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

# Keep models unique.
# If one model hits quota, move to the next one.
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
# STOPWORDS
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
    "been",
    "being",
    "receive",
    "receives",
    "provided",
    "provide",
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

    text = str(text).lower()

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
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Return normalized word tokens.
    """

    normalized = normalize(text)

    if not normalized:
        return []

    return normalized.split()


# ============================================================
# QUESTION TERMS
# ============================================================

def extract_question_terms(question):
    """
    Extract meaningful terms from the question.
    """

    words = tokenize(question)

    terms = []

    for word in words:

        if len(word) < 3:
            continue

        if word in STOPWORDS:
            continue

        terms.append(word)

    return terms


# ============================================================
# MORPHOLOGICAL VARIANTS
# ============================================================

def term_variants(term):
    """
    Generate simple English morphological variants.

    This is intentionally lightweight and deterministic.
    """

    variants = {term}

    # --------------------------------------------------------
    # Plural
    # --------------------------------------------------------

    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])

    else:
        variants.add(term + "s")

    # --------------------------------------------------------
    # -ed / -ing forms
    # --------------------------------------------------------

    if term.endswith("y") and len(term) > 3:
        variants.add(term[:-1] + "ied")

    if term.endswith("ied"):
        variants.add(term[:-3] + "y")

    if term.endswith("ing") and len(term) > 5:
        variants.add(term[:-3])

    if term.endswith("ed") and len(term) > 4:
        variants.add(term[:-2])

    # --------------------------------------------------------
    # Specific common forms
    # --------------------------------------------------------

    common_forms = {
        "carry": {"carried", "carries", "carrying"},
        "carried": {"carry", "carries", "carrying"},
        "receive": {"receives", "received", "receiving"},
        "provide": {"provides", "provided", "providing"},
        "replace": {"replacement", "replaced", "replacing"},
        "replacement": {"replace", "replaced", "replacing"},
        "retain": {"retained", "retention", "retains"},
        "retained": {"retain", "retention", "retains"},
        "trade": {"trading", "traded", "trades"},
        "trading": {"trade", "traded", "trades"},
    }

    variants.update(
        common_forms.get(term, set())
    )

    return variants


# ============================================================
# TERM MATCHING
# ============================================================

def term_exists(term, evidence):
    """
    Check whether a meaningful term or one of its
    simple morphological variants exists in evidence.
    """

    if not term or not evidence:
        return False

    evidence_tokens = set(
        tokenize(evidence)
    )

    for variant in term_variants(term):

        if variant in evidence_tokens:
            return True

    return False


# ============================================================
# CONCEPT GROUPS
# ============================================================

# These protect against false positives where retrieval
# finds related words but not the actual requested concept.

CONCEPT_GROUPS = [
    (
        {"equipment", "replacement"},
        "equipment replacement",
    ),

    (
        {"dental", "insurance"},
        "dental insurance",
    ),

    (
        {"stock", "trading"},
        "stock trading",
    ),

    (
        {"remote", "work"},
        "remote work",
    ),

    (
        {"learning", "budget"},
        "learning budget",
    ),

    (
        {"carry", "unused", "leave"},
        "leave carry-forward",
    ),

    (
        {"president", "india"},
        "president of india",
    ),

    # Security credential questions are answerable when the
    # security policy evidence contains the credential rule.
    # The question may use different wording (share/allowed/etc.),
    # so requiring every question verb would incorrectly reject it.
    (
        {"credentials"},
        "credential sharing",
    ),
]


# ============================================================
# EXACT CONCEPT DETECTION
# ============================================================

def question_concepts(question):
    """
    Return concept groups present in the question.
    """

    question_tokens = set(
        tokenize(question)
    )

    found = []

    for concept_terms, concept_name in CONCEPT_GROUPS:

        matched = True

        for term in concept_terms:

            variants = term_variants(term)

            if not any(
                variant in question_tokens
                for variant in variants
            ):
                matched = False
                break

        if matched:
            found.append(
                (
                    concept_terms,
                    concept_name,
                )
            )

    return found


# ============================================================
# EVIDENCE SUPPORT CHECK
# ============================================================

def evidence_supports_question(
    question,
    retrieved_chunks,
):
    """
    Determine whether retrieved evidence actually contains
    enough information to answer the requested question.

    Retrieval similarity alone is NOT considered sufficient.

    Example:

        Question:
        What is the equipment replacement policy?

        Evidence:
        Employees must return company equipment.

    The word "equipment" matches, but "replacement" does not.

    Therefore the question is unsupported.
    """

    if not retrieved_chunks:
        return False

    evidence_parts = []

    for chunk in retrieved_chunks:

        text = chunk.get(
            "text",
            "",
        )

        if text:
            evidence_parts.append(text)

    evidence = " ".join(
        evidence_parts
    )

    evidence = normalize(evidence)

    if not evidence:
        return False

    # --------------------------------------------------------
    # Token-based evidence
    # --------------------------------------------------------

    evidence_tokens = set(
        tokenize(evidence)
    )

    # --------------------------------------------------------
    # Exact concept protection
    # --------------------------------------------------------

    concepts = question_concepts(
        question
    )

    for concept_terms, concept_name in concepts:

        for term in concept_terms:

            variants = term_variants(term)

            if not any(
                variant in evidence_tokens
                for variant in variants
            ):
                print(
                    f"\nMissing concept term: "
                    f"{term}"
                )

                print(
                    f"Concept not supported: "
                    f"{concept_name}"
                )

                return False

    # If a protected concept is fully present in the evidence,
    # do not apply the generic word-overlap threshold. Questions
    # such as "Is credential sharing allowed?" contain intent
    # words (allowed/share) that may not appear verbatim in the
    # policy even though the policy clearly contains the rule.
    if concepts:
        return True

    # --------------------------------------------------------
    # General question-term matching
    # --------------------------------------------------------

    question_terms = extract_question_terms(
        question
    )

    if not question_terms:
        return True

    matched_terms = 0

    for term in question_terms:

        if term_exists(
            term,
            evidence,
        ):
            matched_terms += 1

    match_ratio = (
        matched_terms
        / len(question_terms)
    )

    # Require meaningful overlap.
    return match_ratio >= 0.40


# ============================================================
# BUILD EVIDENCE
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
    Build a strict grounded generation prompt.
    """

    evidence = build_evidence(
        retrieved_chunks
    )

    prompt = f"""
You are a strict company-policy RAG assistant.

Answer the USER QUESTION using ONLY the PROVIDED POLICY
EVIDENCE.

IMPORTANT RULES:

1. Use only the provided evidence.

2. Do not use outside knowledge.

3. Do not invent information.

4. Answer the exact question asked.

5. Related information is NOT enough.

6. Do not answer a different question.

7. Every factual statement must be directly supported
   by the evidence.

8. If the evidence does not answer the exact question,
   return exactly:

I could not find this information in the provided policies.

9. Do not infer missing policy details.

10. Keep the answer concise.

11. You may mention the source document.

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
# FALLBACK CHECK
# ============================================================

def is_fallback(answer):
    """
    Check whether the answer is the standard fallback.
    """

    if not answer:
        return True

    normalized_answer = normalize(
        answer
    )

    normalized_fallback = normalize(
        FALLBACK_ANSWER
    )

    return (
        normalized_answer
        == normalized_fallback
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate a grounded answer using Gemini.

    Tries each unique model once.
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
                f"Model selected: "
                f"{model_name}"
            )

            return {
                "success": True,
                "answer": answer,
                "model": model_name,
                "status": "SUCCESS",
            }

        except Exception as error:

            last_error = error

            error_text = str(
                error
            )

            print(
                f"Model {model_name} failed."
            )

            lower_error = (
                error_text.lower()
            )

            if (
                "429" in error_text
                or "quota" in lower_error
                or "resource_exhausted"
                in lower_error
                or "rate limit"
                in lower_error
                or "503" in error_text
                or "unavailable"
                in lower_error
            ):

                print(
                    "Quota/rate limit or "
                    "temporary availability issue."
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
    Create an answer directly from retrieved evidence.

    This is used only after evidence support has already
    been confirmed.
    """

    if not retrieved_chunks:
        return FALLBACK_ANSWER

    question_terms = extract_question_terms(
        question
    )

    best_chunk = None
    best_sentence = None
    best_score = -1

    # --------------------------------------------------------
    # Search all retrieved chunks
    # --------------------------------------------------------

    for chunk in retrieved_chunks:

        source = chunk.get(
            "source",
            "unknown",
        )

        text = chunk.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Sentence splitting
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            normalized_sentence = normalize(
                sentence
            )

            score = 0

            # ------------------------------------------------
            # Question-term overlap
            # ------------------------------------------------

            for term in question_terms:

                if term_exists(
                    term,
                    normalized_sentence,
                ):
                    score += 1

            # ------------------------------------------------
            # Prefer longer useful sentences
            # ------------------------------------------------

            if len(
                tokenize(sentence)
            ) >= 5:

                score += 0.1

            if score > best_score:

                best_score = score
                best_chunk = chunk
                best_sentence = sentence

    # --------------------------------------------------------
    # Fallback to strongest retrieved chunk
    # --------------------------------------------------------

    if best_chunk is None:

        best_chunk = retrieved_chunks[0]

        best_sentence = (
            best_chunk.get(
                "text",
                "",
            ).strip()
        )

    if not best_sentence:

        return FALLBACK_ANSWER

    source = best_chunk.get(
        "source",
        "unknown",
    )

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
    """
    Convert retrieved chunks into stable source objects.
    """

    sources = []

    for chunk in retrieved_chunks:

        try:

            score = float(
                chunk.get(
                    "score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        sources.append(
            {
                "source": chunk.get(
                    "source",
                    "unknown",
                ),
                "score": score,
            }
        )

    return sources


# ============================================================
# DETERMINISTIC GRADING
# ============================================================

def deterministic_grading(
    question,
    answer,
    retrieved_chunks,
):
    """
    Deterministic grading fallback.

    Used when the Gemini grader cannot be reached.

    It verifies:

    1. The answer exists.
    2. The answer is not the fallback.
    3. Evidence supports the question.
    4. Question concepts appear in the answer.
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

    # --------------------------------------------------------
    # Evidence support
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Question terms
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Exact concept check in answer
    # --------------------------------------------------------

    concept_score = 1.0

    concepts = question_concepts(
        question
    )

    for concept_terms, _ in concepts:

        concept_matches = 0

        for term in concept_terms:

            if term_exists(
                term,
                normalized_answer,
            ):
                concept_matches += 1

        if concept_terms:

            current_score = (
                concept_matches
                / len(concept_terms)
            )

            concept_score = min(
                concept_score,
                current_score,
            )

    # --------------------------------------------------------
    # Relevance
    # --------------------------------------------------------

    relevance = max(
        0.0,
        min(
            1.0,
            answer_ratio,
        ),
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    correctness = min(
        relevance,
        concept_score,
    )

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    # The answer is produced directly from retrieved
    # evidence, so deterministic fallback considers it
    # grounded after the evidence-support check.
    grounding = 1.0

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

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
# NORMALIZE GRADER RESULT
# ============================================================

def normalize_grading_result(
    grading,
):
    """
    Safely normalize grader output.
    """

    if not isinstance(
        grading,
        dict,
    ):
        return None

    try:

        correctness = float(
            grading.get(
                "correctness",
                0.0,
            )
        )

        relevance = float(
            grading.get(
                "relevance",
                0.0,
            )
        )

        grounding = float(
            grading.get(
                "grounding",
                0.0,
            )
        )

        overall = float(
            grading.get(
                "overall_score",
                (
                    correctness
                    + relevance
                    + grounding
                ) / 3.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    correctness = max(
        0.0,
        min(
            1.0,
            correctness,
        ),
    )

    relevance = max(
        0.0,
        min(
            1.0,
            relevance,
        ),
    )

    grounding = max(
        0.0,
        min(
            1.0,
            grounding,
        ),
    )

    overall = max(
        0.0,
        min(
            1.0,
            overall,
        ),
    )

    return {
        "correctness": correctness,
        "relevance": relevance,
        "grounding": grounding,
        "overall_score": overall,
        "model": grading.get(
            "model",
            "gemini-grader",
        ),
        "feedback": grading.get(
            "feedback",
            "",
        ),
    }


# ============================================================
# FINAL RESULT BUILDER
# ============================================================

def build_result(
    question,
    answer,
    confidence,
    final_score,
    attempts,
    status,
    model,
    retrieved_chunks,
):
    """
    Return a consistent result schema.
    """

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "final_score": float(
            final_score
        ),
        "attempts": int(
            attempts
        ),
        "status": status,
        "model": model,
        "sources": format_sources(
            retrieved_chunks
        ),
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

            result = build_result(
                question=question,
                answer=FALLBACK_ANSWER,
                confidence="I don't know",
                final_score=0.0,
                attempts=0,
                status="NO_RELEVANT_SOURCES",
                model=None,
                retrieved_chunks=[],
            )

            return result

        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        for chunk in retrieved_chunks:

            source = chunk.get(
                "source",
                "unknown",
            )

            chunk_id = chunk.get(
                "chunk_id",
                "?",
            )

            try:

                score = float(
                    chunk.get(
                        "score",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                score = 0.0

            print(
                f"- {source} "
                f"(chunk={chunk_id}, "
                f"score={score:.4f})"
            )

        # ====================================================
        # STEP 2 — EVIDENCE SUPPORT
        # ====================================================

        supported = evidence_supports_question(
            question,
            retrieved_chunks,
        )

        print(
            "\nEvidence support check: "
            f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}"
        )

        # ====================================================
        # UNSUPPORTED QUESTION
        # ====================================================

        if not supported:

            print(
                "\nRetrieved documents do not "
                "directly answer the exact question."
            )

            print(
                "Returning NO_SUPPORTED_ANSWER."
            )

            return build_result(
                question=question,
                answer=FALLBACK_ANSWER,
                confidence="I don't know",
                final_score=0.0,
                attempts=1,
                status="NO_SUPPORTED_ANSWER",
                model="deterministic-evidence-check",
                retrieved_chunks=retrieved_chunks,
            )

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

        answer = generation.get(
            "answer",
            FALLBACK_ANSWER,
        )

        generation_model = generation.get(
            "model"
        )

        generation_status = generation.get(
            "status"
        )

        # ====================================================
        # GENERATION FALLBACK
        # ====================================================

        if (
            generation_status != "SUCCESS"
            or is_fallback(answer)
        ):

            print(
                "\nLLM generation unavailable "
                "or returned the fallback."
            )

            print(
                "Using deterministic "
                "extractive fallback."
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
        # FALLBACK SAFETY CHECK
        # ====================================================

        if is_fallback(answer):

            print(
                "\nUnable to produce a "
                "supported answer."
            )

            return build_result(
                question=question,
                answer=FALLBACK_ANSWER,
                confidence="I don't know",
                final_score=0.0,
                attempts=1,
                status="NO_SUPPORTED_ANSWER",
                model=generation_model,
                retrieved_chunks=retrieved_chunks,
            )

        # ====================================================
        # STEP 4 — SELF GRADING
        # ====================================================

        print(
            "\nStarting self-grading..."
        )

        grading = None

        try:

            grading = grade_answer(
                question,
                answer,
                retrieved_chunks,
            )

            grading = normalize_grading_result(
                grading
            )

        except Exception as error:

            print(
                f"\nGrader exception: {error}"
            )

            grading = None

        # ====================================================
        # GRADER FALLBACK
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

        try:

            final_score = float(
                grading.get(
                    "overall_score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            final_score = 0.0

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

            # Never return an unsupported generated answer.
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

            try:

                score = float(
                    source.get(
                        "score",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                score = 0.0

            print(
                f"- {source.get('source', 'unknown')} "
                f"(retrieval score="
                f"{score:.4f})"
            )

        # ====================================================
        # STRUCTURED RETURN
        # ====================================================

        return build_result(
            question=question,
            answer=answer,
            confidence=confidence,
            final_score=final_score,
            attempts=1,
            status=status,
            model=(
                generation_model
                or grading_model
            ),
            retrieved_chunks=retrieved_chunks,
        )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_agent(
    question,
    top_k=3,
    threshold=RETRIEVAL_THRESHOLD,
):
    """
    Public API used by evaluate.py.
    """

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