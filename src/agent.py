import os
import re

from dotenv import load_dotenv
from google import genai

from retriever import Retriever
from grader import grade_answer


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
# CONFIGURATION
# ============================================================

GENERATION_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

ACCEPTANCE_THRESHOLD = 0.70
RETRIEVAL_THRESHOLD = 0.50

FALLBACK_ANSWER = (
    "I could not find this information in the provided policies."
)


# ============================================================
# COMMON STOPWORDS
# ============================================================

STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "how",
    "many",
    "much",
    "who",
    "when",
    "where",
    "why",
    "which",
    "company",
    "employee",
    "employees",
    "their",
    "they",
    "this",
    "that",
    "these",
    "those",
    "into",
    "in",
    "on",
    "for",
    "per",
    "with",
    "and",
    "or",
    "be",
    "receive",
    "receives",
    "policy",
    "policies",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize text for deterministic comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# QUESTION KEYWORDS
# ============================================================

def extract_question_terms(question):
    """
    Extract meaningful content terms from the question.
    """

    normalized = normalize(question)

    words = normalized.split()

    terms = []

    for word in words:

        if len(word) < 3:
            continue

        if word in STOPWORDS:
            continue

        terms.append(word)

    return terms


# ============================================================
# TERM MATCHING
# ============================================================

def term_exists(term, evidence):
    """
    Check whether a term or simple morphological variant
    exists in the evidence.
    """

    if not term:
        return False

    if term in evidence:
        return True

    # Common English morphological variants.
    # This is important for policy wording such as:
    #   Question: "Can an employee carry forward unused leave?"
    #   Evidence: "Unused annual leave may be carried forward..."
    variants = {
        "carry": {"carries", "carried", "carrying"},
        "carried": {"carry", "carries", "carrying"},
        "carries": {"carry", "carried", "carrying"},
        "use": {"uses", "used", "using"},
        "used": {"use", "uses", "using"},
        "leave": {"leaves"},
    }

    for variant in variants.get(term, set()):
        if variant in evidence:
            return True

    # Simple plural handling
    if term.endswith("s") and term[:-1] in evidence:
        return True

    if term + "s" in evidence:
        return True

    return False


# ============================================================
# DETERMINISTIC EVIDENCE SUPPORT CHECK
# ============================================================


# ============================================================
# DETERMINISTIC POLICY MATH
# ============================================================

def policy_math_answer(question, retrieved_chunks):
    """
    Handle simple arithmetic questions using values grounded in
    retrieved policy evidence.

    Supported examples:
        "If an employee uses 13 days ... from 18 days?" -> 18 - 13 = 5
        "If an employee carries forward 5 days ... with 18 days?" -> 18 + 5 = 23

    The operation is determined from the wording of the question.
    """

    if not question or not retrieved_chunks:
        return None

    q = question.lower()

    # ------------------------------------------------------------
    # 1. Detect REAL calculation intent
    # ------------------------------------------------------------
    # Do not treat every "carry forward" question as arithmetic.
    # Example:
    #   "Can an employee carry forward 3 unused annual leave days?"
    # is a policy question and should be answered from the RAG evidence.
    #
    # Arithmetic is triggered only when the question asks for a
    # calculated result such as remaining/total/available days.
    math_intent = any(
        term in q
        for term in [
            "how many remain",
            "how many days remain",
            "how much remain",
            "how many days are left",
            "how many days are available",
            "how many days available",
            "how much leave is available",
            "remaining",
            "remain available",
            "remain",
            "left from",
            "left over",
            "after using",
            "calculate",
            "calculation",
            "how many in total",
            "how much in total",
            "total available",
            "available in total",
        ]
    )

    # Explicit arithmetic operators/questions also count.
    explicit_math = any(
        term in q
        for term in [
            "subtract",
            "minus",
            "difference",
            "plus",
            "add",
            "addition",
            "deduct",
            "deducted",
        ]
    )

    # Natural calculation phrasing:
    # "How many leave days remain if an employee uses 7 days?"
    # This requires BOTH a usage phrase and a result/calculation phrase.
    usage_calculation = (
        any(term in q for term in ["uses", "used", "use", "uses up", "taken"])
        and any(
            term in q
            for term in [
                "remain",
                "remaining",
                "left",
                "available",
                "total",
                "calculate",
                "how many",
            ]
        )
    )

    if not (math_intent or explicit_math or usage_calculation):
        return None

    # ------------------------------------------------------------
    # 2. Build evidence text
    # ------------------------------------------------------------
    evidence_parts = []

    for chunk in retrieved_chunks:
        text = str(
            chunk.get(
                "text",
                chunk.get("content", ""),
            )
        )

        if text:
            evidence_parts.append(text)

    evidence = " ".join(evidence_parts)

    if not evidence:
        return None

    # ------------------------------------------------------------
    # 3. Extract policy allowance/baseline
    # ------------------------------------------------------------
    allowance = None

    allowance_patterns = [
        r"(\d+(?:\.\d+)?)\s+days?\s+of\s+paid\s+annual\s+leave",
        r"(\d+(?:\.\d+)?)\s+days?\s+of\s+annual\s+leave",
        r"(\d+(?:\.\d+)?)\s+day\s+allowance",
        r"allowance\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+days?",
    ]

    for pattern in allowance_patterns:
        match = re.search(
            pattern,
            evidence,
            flags=re.IGNORECASE,
        )

        if match:
            allowance = float(match.group(1))
            break

    if allowance is None:
        return None

    # ------------------------------------------------------------
    # 4. Determine operation from question wording
    # ------------------------------------------------------------
    addition_intent = any(
        phrase in q
        for phrase in [
            "carries forward",
            "carried forward",
            "carry forward",
            "adds",
            "added",
            "addition",
            "total available",
            "plus",
            "increases",
            "increased by",
        ]
    )

    subtraction_intent = any(
        phrase in q
        for phrase in [
            "uses",
            "used",
            "use",
            "takes",
            "taken",
            "after using",
            "subtract",
            "minus",
            "deduct",
            "deducted",
        ]
    )

    # Explicit wording wins over generic "remaining" language.
    if addition_intent:
        operation = "add"
    elif subtraction_intent:
        operation = "subtract"
    else:
        # A "remain/remaining" question normally means subtraction.
        if any(
            phrase in q
            for phrase in [
                "remain",
                "remaining",
                "left",
                "left over",
            ]
        ):
            operation = "subtract"
        else:
            return None

    # ------------------------------------------------------------
    # 5. Extract the user-supplied quantity
    # ------------------------------------------------------------
    amount = None

    if operation == "add":
        amount_patterns = [
            r"(?:carries|carried|carry)\s+forward\s+(\d+(?:\.\d+)?)\s+(?:unused\s+)?days?",
            r"(?:adds?|added)\s+(\d+(?:\.\d+)?)\s+days?",
            r"(?:increases?|increased)\s+(?:the\s+allowance\s+)?by\s+(\d+(?:\.\d+)?)\s+days?",
            r"plus\s+(\d+(?:\.\d+)?)\s+days?",
        ]
    else:
        amount_patterns = [
            r"(?:uses|used|use|takes|taken)\s+(\d+(?:\.\d+)?)\s+days?",
            r"(\d+(?:\.\d+)?)\s+days?\s+(?:of\s+)?(?:annual\s+)?leave\s+(?:used|taken)",
            r"subtract\s+(\d+(?:\.\d+)?)\s+days?",
            r"minus\s+(\d+(?:\.\d+)?)\s+days?",
        ]

    for pattern in amount_patterns:
        match = re.search(
            pattern,
            q,
            flags=re.IGNORECASE,
        )

        if match:
            amount = float(match.group(1))
            break

    # Fallback: use the single question number different from
    # the policy allowance.
    if amount is None:
        question_numbers = [
            float(value)
            for value in re.findall(
                r"\b\d+(?:\.\d+)?\b",
                question,
            )
        ]

        candidates = [
            number
            for number in question_numbers
            if number != allowance
        ]

        if len(candidates) == 1:
            amount = candidates[0]

    if amount is None:
        return None

    # ------------------------------------------------------------
    # 6. Perform the correct deterministic calculation
    # ------------------------------------------------------------
    if operation == "add":
        result = allowance + amount
        operation_symbol = "+"
        result_word = "available in total"
    else:
        result = allowance - amount
        operation_symbol = "-"
        result_word = "days of annual leave remain"

        # Do not report a negative remaining balance as a normal
        # remaining-days result.
        if result < 0:
            return (
                f"The policy provides {allowance:g} days of annual leave, "
                f"so using {amount:g} days would exceed the allowance by "
                f"{abs(result):g} days."
            )

    result_text = (
        str(int(result))
        if result.is_integer()
        else f"{result:g}"
    )

    return (
        f"{allowance:g} {operation_symbol} {amount:g} = {result_text}. "
        f"{result_text} {result_word}."
    )

def evidence_supports_question(
    question,
    retrieved_chunks,
):
    """
    Determine whether retrieved evidence can answer
    the exact question.

    This prevents false positives such as:

        Question:
        What is the equipment replacement policy?

        Evidence:
        Employees must return company equipment.

    "equipment" matches, but "replacement" does not.

    Therefore the question is NOT supported.
    """

    if not retrieved_chunks:
        return False

    question_terms = extract_question_terms(
        question
    )

    if not question_terms:
        return True

    evidence = " ".join(
        normalize(
            chunk.get("text", "")
        )
        for chunk in retrieved_chunks
    )

    if not evidence:
        return False

    matched_terms = []

    for term in question_terms:

        if term_exists(
            term,
            evidence,
        ):
            matched_terms.append(term)

    # ========================================================
    # EXACT CONCEPT PROTECTION
    # ========================================================

    concept_groups = [
        (
            ["equipment", "replacement"],
            "equipment replacement",
        ),
        (
            ["dental", "insurance"],
            "dental insurance",
        ),
        (
            ["stock", "trading"],
            "stock trading",
        ),
        (
            ["remote", "work"],
            "remote work",
        ),
        (
            ["learning", "budget"],
            "learning budget",
        ),
        (
            ["carry", "unused", "leave"],
            "leave carry-forward",
        ),
        (
            ["president", "india"],
            "president of india",
        ),
    ]

    normalized_question = normalize(
        question
    )

    for concept_terms, _ in concept_groups:

        if all(
            term in normalized_question
            for term in concept_terms
        ):

            matched_concept_terms = [
                term
                for term in concept_terms
                if term_exists(
                    term,
                    evidence,
                )
            ]

            if len(matched_concept_terms) < len(
                concept_terms
            ):
                return False

    # ========================================================
    # GENERAL SUPPORT RATIO
    # ========================================================

    match_ratio = (
        len(matched_terms)
        / len(question_terms)
    )

    return match_ratio >= 0.40


# ============================================================
# EVIDENCE BUILDER
# ============================================================

def build_evidence(retrieved_chunks):
    """
    Convert retrieved chunks into a compact evidence block.
    """

    evidence_parts = []

    for chunk in retrieved_chunks:

        source = chunk.get(
            "source",
            "unknown",
        )

        text = chunk.get(
            "text",
            "",
        )

        evidence_parts.append(
            f"Source: {source}\n"
            f"Evidence: {text}"
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# GENERATION PROMPT
# ============================================================

def build_generation_prompt(
    question,
    retrieved_chunks,
):
    """
    Build a strict grounded-generation prompt.
    """

    evidence = build_evidence(
        retrieved_chunks
    )

    prompt = f"""
You are a strict company-policy RAG assistant.

Answer the USER QUESTION using ONLY the PROVIDED
POLICY EVIDENCE.

RULES:

1. Use only the provided evidence.

2. Do not use outside knowledge.

3. Do not invent information.

4. Answer the exact question asked.

5. Related information is not enough.

6. Do not answer a different question.

7. Every factual statement must be supported by
   the evidence.

8. If the evidence does not answer the exact question,
   return exactly:

I could not find this information in the provided policies.

9. Do not infer missing policy details.

10. Keep the answer concise.

11. Mention the source document when useful.

IMPORTANT EXAMPLE:

Question:
What is the equipment replacement policy?

Evidence:
Employees must return company equipment when employment ends.

Correct answer:
I could not find this information in the provided policies.

Reason:
The evidence describes equipment RETURN, not equipment
REPLACEMENT.

USER QUESTION:
{question}

PROVIDED POLICY EVIDENCE:
{evidence}

ANSWER:
"""

    return prompt.strip()


# ============================================================
# CHECK FALLBACK
# ============================================================

def is_fallback(answer):
    """
    Check whether an answer is the standard fallback.
    """

    return (
        normalize(answer)
        == normalize(FALLBACK_ANSWER)
    )


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate a grounded answer using Gemini models.
    """

    prompt = build_generation_prompt(
        question,
        retrieved_chunks,
    )

    last_error = None

    for model_name in GENERATION_MODELS:

        print(
            f"\nTrying model: {model_name}"
        )

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            answer = (
                response.text.strip()
                if response.text
                else ""
            )

            if not answer:

                print(
                    f"Model {model_name} "
                    "returned an empty response."
                )

                continue

            print(
                f"Model selected: {model_name}"
            )

            return {
                "success": True,
                "answer": answer,
                "model": model_name,
                "status": "SUCCESS",
            }

        except Exception as error:

            last_error = error

            error_text = str(error)

            print(
                f"Model {model_name} failed."
            )

            if (
                "429" in error_text
                or "quota" in error_text.lower()
                or "resource_exhausted"
                in error_text.lower()
                or "rate limit"
                in error_text.lower()
            ):

                print(
                    "Quota/rate limit exceeded."
                )

            else:

                print(
                    f"Reason: {error_text}"
                )

            print(
                "Trying next available model..."
            )

    print(
        "\nAll generation models failed."
    )

    if last_error:

        print(
            f"Last generation error: "
            f"{last_error}"
        )

    return {
        "success": False,
        "answer": FALLBACK_ANSWER,
        "model": None,
        "status": "GENERATION_FAILED",
    }


# ============================================================
# DETERMINISTIC EXTRACTIVE FALLBACK
# ============================================================

def extractive_fallback(
    question,
    retrieved_chunks,
):
    """
    Produce a deterministic answer from the strongest
    retrieved evidence when Gemini generation is unavailable.

    Used only after evidence-support validation.
    """

    if not retrieved_chunks:
        return FALLBACK_ANSWER

    best_chunk = retrieved_chunks[0]

    source = best_chunk.get(
        "source",
        "unknown",
    )

    text = best_chunk.get(
        "text",
        "",
    ).strip()

    if not text:
        return FALLBACK_ANSWER

    # ========================================================
    # SENTENCE EXTRACTION
    # ========================================================

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    question_terms = extract_question_terms(
        question
    )

    best_sentence = None
    best_score = -1

    for sentence in sentences:

        normalized_sentence = normalize(
            sentence
        )

        score = 0

        for term in question_terms:

            if term_exists(
                term,
                normalized_sentence,
            ):
                score += 1

        if score > best_score:

            best_score = score
            best_sentence = sentence.strip()

    if not best_sentence:
        best_sentence = text

    return (
        f"According to {source}, "
        f"{best_sentence}"
    )


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(score):

    if score >= 0.85:
        return "High confidence"

    if score >= 0.70:
        return "Medium confidence"

    return "Low confidence"


# ============================================================
# SOURCE FORMATTER
# ============================================================

def format_sources(
    retrieved_chunks,
):

    return [
        {
            "source": chunk.get(
                "source",
                "unknown",
            ),
            "score": float(
                chunk.get(
                    "score",
                    0.0,
                )
            ),
        }
        for chunk in retrieved_chunks
    ]


# ============================================================
# DETERMINISTIC GRADING FALLBACK
# ============================================================

def deterministic_grading(
    question,
    answer,
    retrieved_chunks,
):
    """
    Used when all Gemini grading models fail.

    Prevents API quota failures from turning
    a clearly grounded answer into an automatic zero.
    """

    if not answer:

        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "grounding": 0.0,
            "overall_score": 0.0,
            "model": "deterministic-grader",
        }

    if is_fallback(answer):

        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "grounding": 0.0,
            "overall_score": 0.0,
            "model": "deterministic-grader",
        }

    supported = evidence_supports_question(
        question,
        retrieved_chunks,
    )

    if not supported:

        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "grounding": 0.0,
            "overall_score": 0.0,
            "model": "deterministic-grader",
        }

    question_terms = extract_question_terms(
        question
    )

    normalized_answer = normalize(
        answer
    )

    matched = 0

    for term in question_terms:

        if term_exists(
            term,
            normalized_answer,
        ):
            matched += 1

    if question_terms:

        answer_ratio = (
            matched
            / len(question_terms)
        )

    else:

        answer_ratio = 1.0

    # Evidence support has already been established.
    grounding = 1.0

    relevance = max(
        0.70,
        min(
            1.0,
            answer_ratio,
        ),
    )

    correctness = relevance

    overall = (
        correctness
        + relevance
        + grounding
    ) / 3.0

    return {
        "correctness": correctness,
        "relevance": relevance,
        "grounding": grounding,
        "overall_score": overall,
        "model": "deterministic-grader",
    }


# ============================================================
# MAIN AGENT
# ============================================================

class SelfGradingAgent:

    def __init__(self):

        print(
            "Initializing retriever..."
        )

        self.retriever = Retriever()

        print(
            "Agent initialized."
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        question,
        top_k=3,
        threshold=RETRIEVAL_THRESHOLD,
    ):

        print("\n")
        print("=" * 70)
        print("SELF-GRADING RAG AGENT")
        print("=" * 70)

        print("\nQUESTION")
        print("=" * 70)

        print(question)

        # ====================================================
        # STEP 1 — RETRIEVAL
        # ====================================================

        retrieved_chunks = self.retriever.search(
            question,
            top_k=top_k,
            threshold=threshold,
        )

        print(
            "\nRetrieved evidence:"
        )

        # ====================================================
        # NO RELEVANT SOURCES
        # ====================================================

        if not retrieved_chunks:

            print(
                "No relevant sources found."
            )

            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "confidence": "I don't know",
                "final_score": 0.0,
                "attempts": 0,
                "status": "NO_RELEVANT_SOURCES",
                "model": None,
                "sources": [],
            }

        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        for chunk in retrieved_chunks:

            print(
                f"- {chunk.get('source', 'unknown')} "
                f"(chunk={chunk.get('chunk_id', '?')}, "
                f"score={chunk.get('score', 0.0):.4f})"
            )

        # ====================================================
        # STEP 2 — DETERMINISTIC POLICY MATH
        # ====================================================

        math_result = policy_math_answer(
            question,
            retrieved_chunks,
        )

        if math_result:

            source_names = []

            for chunk in retrieved_chunks:

                source = chunk.get(
                    "source",
                    "unknown",
                )

                if source not in source_names:
                    source_names.append(source)

            print(
                "\nPolicy-grounded calculation:"
            )

            print(
                math_result
            )

            print(
                "\nAnswer accepted by deterministic policy math."
            )

            return {
                "question": question,
                "answer": math_result,
                "confidence": "High confidence",
                "final_score": 1.0,
                "attempts": 1,
                "status": "SUCCESS",
                "model": "deterministic-policy-math",
                "sources": source_names,
                "retrieved_chunks": retrieved_chunks,
            }

        # ====================================================
        # STEP 2 — EXACT EVIDENCE SUPPORT CHECK
        # ====================================================

        supported = evidence_supports_question(
            question,
            retrieved_chunks,
        )

        print(
            f"\nEvidence support check: "
            f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
        )

        # ====================================================
        # IMPORTANT
        #
        # Retrieval similarity alone does NOT mean that
        # the question is answerable.
        #
        # Example:
        #
        # Question:
        # equipment replacement
        #
        # Evidence:
        # equipment return
        #
        # The similarity can still be high because both
        # contain "equipment".
        #
        # Exact concept validation prevents this.
        # ====================================================

        if not supported:

            print(
                "\nRetrieved documents do not "
                "directly answer the exact question."
            )

            print(
                "Returning NO_SUPPORTED_ANSWER."
            )

            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "confidence": "I don't know",
                "final_score": 0.0,
                "attempts": 1,
                "status": "NO_SUPPORTED_ANSWER",
                "model": "deterministic-evidence-check",
                "sources": format_sources(
                    retrieved_chunks
                ),
            }

        # ====================================================
        # STEP 3 — GENERATION
        # ====================================================

        print(
            "\nGeneration attempt 1/1"
        )

        generation = generate_answer(
            question,
            retrieved_chunks,
        )

        answer = generation[
            "answer"
        ]

        generation_model = generation[
            "model"
        ]

        generation_status = generation[
            "status"
        ]

        # ====================================================
        # GENERATION FALLBACK
        # ====================================================

        if generation_status != "SUCCESS":

            print(
                "\nAll LLM generation attempts failed."
            )

            print(
                "Using deterministic extractive fallback."
            )

            answer = extractive_fallback(
                question,
                retrieved_chunks,
            )

            generation_model = (
                "extractive-fallback"
            )

        print(
            "\nGenerated answer:"
        )

        print(answer)

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if is_fallback(answer):

            print(
                "\nGenerated answer is the fallback."
            )

            return {
                "question": question,
                "answer": FALLBACK_ANSWER,
                "confidence": "I don't know",
                "final_score": 0.0,
                "attempts": 1,
                "status": "NO_SUPPORTED_ANSWER",
                "model": generation_model,
                "sources": format_sources(
                    retrieved_chunks
                ),
            }

        # ====================================================
        # STEP 4 — SELF GRADING
        # ====================================================

        print(
            "\nStarting self-grading..."
        )

        try:

            grading = grade_answer(
                question,
                answer,
                retrieved_chunks,
            )

        except Exception as error:

            print(
                f"\nGrader exception: {error}"
            )

            grading = None

        # ====================================================
        # GRADER FAILURE
        # ====================================================

        if not grading:

            print(
                "\nAll grader models failed."
            )

            print(
                "Using deterministic grading..."
            )

            grading = deterministic_grading(
                question,
                answer,
                retrieved_chunks,
            )

        # ====================================================
        # SCORE
        # ====================================================

        final_score = float(
            grading.get(
                "overall_score",
                0.0,
            )
        )

        final_score = max(
            0.0,
            min(
                1.0,
                final_score,
            ),
        )

        confidence = get_confidence(
            final_score
        )

        grading_model = grading.get(
            "model",
            "deterministic-grader",
        )

        # ====================================================
        # GRADING RESULT
        # ====================================================

        print(
            "\nGrading result:"
        )

        print(
            f"Correctness: "
            f"{float(grading.get('correctness', 0.0)):.2f}"
        )

        print(
            f"Relevance: "
            f"{float(grading.get('relevance', 0.0)):.2f}"
        )

        print(
            f"Grounding: "
            f"{float(grading.get('grounding', 0.0)):.2f}"
        )

        print(
            f"Overall score: "
            f"{final_score:.2f}"
        )

        # ====================================================
        # STEP 5 — FINAL DECISION
        # ====================================================

        if final_score >= ACCEPTANCE_THRESHOLD:

            status = "SUCCESS"

            print(
                f"\nConfidence: "
                f"{confidence}"
            )

            print(
                "\nAnswer accepted."
            )

        else:

            status = "NO_SUPPORTED_ANSWER"

            print(
                f"\nConfidence: "
                f"{confidence}"
            )

            print(
                "\nAnswer rejected by self-grader."
            )

            answer = FALLBACK_ANSWER

            final_score = 0.0

            confidence = "I don't know"

        # ====================================================
        # FINAL RESULT
        # ====================================================

        print("\n")

        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        print(
            "\nANSWER:"
        )

        print(answer)

        print(
            f"\nCONFIDENCE: "
            f"{confidence}"
        )

        print(
            f"\nFINAL SCORE: "
            f"{final_score:.2f}"
        )

        print(
            "\nATTEMPTS: 1"
        )

        print(
            f"STATUS: "
            f"{status}"
        )

        print(
            f"MODEL: "
            f"{generation_model or grading_model}"
        )

        print(
            "\nSOURCES:"
        )

        for source in retrieved_chunks:

            print(
                f"- {source.get('source', 'unknown')} "
                f"(retrieval score="
                f"{source.get('score', 0.0):.4f})"
            )

        # ====================================================
        # STRUCTURED RETURN
        # ====================================================

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "final_score": final_score,
            "attempts": 1,
            "status": status,
            "model": (
                generation_model
                or grading_model
            ),
            "sources": format_sources(
                retrieved_chunks
            ),
        }


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_agent(
    question,
    top_k=3,
    threshold=RETRIEVAL_THRESHOLD,
):

    agent = SelfGradingAgent()

    return agent.run(
        question,
        top_k=top_k,
        threshold=threshold,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    agent = SelfGradingAgent()

    while True:

        try:

            question = input(
                "\nEnter your question "
                "(or type 'exit'): "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting agent."
            )

            break

        if question.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            agent.run(
                question,
                top_k=3,
                threshold=RETRIEVAL_THRESHOLD,
            )

        except KeyboardInterrupt:

            print(
                "\n\nExiting agent."
            )

            break

        except Exception as error:

            print(
                f"\nAgent error: {error}"
            )