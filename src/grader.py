import json
import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Add it to the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# MODEL FALLBACK CHAIN
# ============================================================

MODEL_LIST = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


# ============================================================
# DEFAULT FAILURE RESULT
# ============================================================

def failure_result(message, status="GRADING_FAILED"):

    return {
        "correctness": 0.0,
        "relevance": 0.0,
        "grounding": 0.0,
        "overall_score": 0.0,
        "feedback": message,
        "unsupported_claims": [],
        "status": status,
    }


# ============================================================
# GRADING PROMPT
# ============================================================

def build_grading_prompt(question, answer, evidence):
    """Create a prompt for evaluating a generated answer."""

    prompt = f"""
You are an evaluation system for a Retrieval-Augmented Generation (RAG)
application.

Evaluate the generated answer using ONLY the provided evidence.

USER QUESTION:
{question}

PROVIDED EVIDENCE:
{evidence}

GENERATED ANSWER:
{answer}

Evaluate the answer on these three dimensions:

1. Correctness
   - Does the answer correctly answer the user's question?

2. Relevance
   - Does the answer directly address the user's question?
   - Does it avoid unnecessary information?

3. Grounding
   - Are all factual claims in the answer supported by the evidence?
   - Penalize unsupported, invented, or contradictory claims.

Give each score from 0.0 to 1.0.

Then calculate:

overall_score = average of correctness, relevance, and grounding.

Also provide:

- brief feedback
- unsupported_claims

IMPORTANT:

If the generated answer contains ANY factual claim that is not
supported by the provided evidence, include that claim inside
"unsupported_claims" and reduce the grounding score.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "correctness": 0.0,
    "relevance": 0.0,
    "grounding": 0.0,
    "overall_score": 0.0,
    "feedback": "Brief explanation",
    "unsupported_claims": []
}}
"""

    return prompt


# ============================================================
# CLEAN GEMINI JSON OUTPUT
# ============================================================

def clean_json_output(raw_output):

    raw_output = raw_output.strip()

    if raw_output.startswith("```"):

        raw_output = raw_output.replace(
            "```json",
            "",
        )

        raw_output = raw_output.replace(
            "```",
            "",
        )

        raw_output = raw_output.strip()

    return raw_output


# ============================================================
# VALIDATE GRADING RESULT
# ============================================================

def validate_result(result):

    required_fields = [
        "correctness",
        "relevance",
        "grounding",
        "overall_score",
        "feedback",
        "unsupported_claims",
    ]

    for field in required_fields:

        if field not in result:
            return False

    try:

        result["correctness"] = float(
            result["correctness"]
        )

        result["relevance"] = float(
            result["relevance"]
        )

        result["grounding"] = float(
            result["grounding"]
        )

        result["overall_score"] = float(
            result["overall_score"]
        )

    except (ValueError, TypeError):

        return False

    return True


# ============================================================
# MAIN GRADER
# ============================================================

def grade_answer(question, answer, retrieved_chunks):
    """
    Evaluate a generated answer against retrieved evidence.

    Uses multiple Gemini models as a fallback chain.
    """

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not retrieved_chunks:

        return failure_result(
            "No evidence was retrieved for evaluation.",
            status="NO_EVIDENCE",
        )


    # --------------------------------------------------------
    # Build evidence
    # --------------------------------------------------------

    evidence = "\n\n".join(
        [
            f"Source: {chunk['source']}\n"
            f"Evidence: {chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )


    # --------------------------------------------------------
    # Build grading prompt
    # --------------------------------------------------------

    prompt = build_grading_prompt(
        question,
        answer,
        evidence,
    )


    # --------------------------------------------------------
    # Try each model
    # --------------------------------------------------------

    last_error = None

    for model_name in MODEL_LIST:

        print(
            f"\nGrader trying model: "
            f"{model_name}"
        )

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            raw_output = response.text.strip()

            raw_output = clean_json_output(
                raw_output
            )


            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                result = json.loads(
                    raw_output
                )

            except json.JSONDecodeError:

                print(
                    f"Invalid JSON from "
                    f"{model_name}."
                )

                last_error = (
                    f"{model_name} returned invalid JSON."
                )

                continue


            # ------------------------------------------------
            # Validate result
            # ------------------------------------------------

            if not validate_result(result):

                print(
                    f"Invalid grading structure "
                    f"from {model_name}."
                )

                last_error = (
                    f"{model_name} returned "
                    f"an invalid grading structure."
                )

                continue


            # ------------------------------------------------
            # Successful grading
            # ------------------------------------------------

            result["model"] = model_name
            result["status"] = "SUCCESS"

            print(
                f"Grading successful using "
                f"{model_name}."
            )

            return result


        except Exception as error:

            error_text = str(error)

            last_error = error_text

            # ------------------------------------------------
            # Detect quota / rate limit
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
                or "rate limit" in error_text.lower()
            ):

                print(
                    f"Model {model_name} "
                    f"quota/rate limit exceeded."
                )

                print(
                    "Trying next grading model..."
                )

                continue


            # ------------------------------------------------
            # Other API error
            # ------------------------------------------------

            print(
                f"Grader error with "
                f"{model_name}:"
            )

            print(error_text)

            continue


    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    print(
        "\nAll grading models failed."
    )

    return failure_result(
        (
            "All configured Gemini grading models "
            "were unavailable. "
            f"Last error: {last_error}"
        ),
        status="RESOURCE_EXHAUSTED",
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "How many paid annual leave days "
        "do full-time employees receive?"
    )


    evidence_chunks = [

        {
            "source": "02_leave_policy.txt",

            "text": (
                "Full-time employees receive 18 days "
                "of paid annual leave per calendar year."
            ),
        }

    ]


    answer = (
        "Full-time employees receive 18 days "
        "of paid annual leave per calendar year, "
        "and they also receive unlimited sick leave."
    )


    result = grade_answer(
        question,
        answer,
        evidence_chunks,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELF-GRADING RESULT"
    )

    print(
        "=" * 80
    )


    print(
        json.dumps(
            result,
            indent=4,
        )
    )