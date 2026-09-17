import os

from dotenv import load_dotenv
from google import genai

from retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Add it to the .env file."
    )


# Models are tried in this order.
MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

NO_ANSWER = (
    "I could not find this information in the provided policies."
)


# ============================================================
# CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# RETRIEVER
# ============================================================

retriever = Retriever()


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question, retrieved_chunks):
    """
    Build a grounded RAG prompt using retrieved evidence.
    """

    evidence = "\n\n".join(
        [
            (
                f"Source: {chunk['source']}\n"
                f"Evidence: {chunk['text']}"
            )
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the provided evidence.

Rules:

1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the evidence does not contain the answer, say:
   "{NO_ANSWER}"
5. Keep the answer concise and factual.
6. Mention the relevant source document when appropriate.
7. Do not mention information that is not supported by the evidence.

USER QUESTION:
{question}

PROVIDED EVIDENCE:
{evidence}

ANSWER:
"""

    return prompt


# ============================================================
# MODEL GENERATION
# ============================================================

def generate_with_fallback(prompt):
    """
    Generate an answer using Gemini with model fallback.

    Returns:
        answer, selected_model
    """

    last_error = None

    for model_name in MODEL_FALLBACKS:

        print(
            f"\nTrying model: {model_name}"
        )

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            if response.text:

                print(
                    f"Model selected: {model_name}"
                )

                return (
                    response.text.strip(),
                    model_name,
                )

        except Exception as error:

            last_error = error

            print(
                f"Model {model_name} failed."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Trying next available model..."
            )

    raise RuntimeError(
        "All Gemini models failed."
    ) from last_error


# ============================================================
# RAG PIPELINE
# ============================================================

def generate_answer(
    question,
    top_k=3,
    threshold=0.50,
):
    """
    Complete RAG pipeline:

        Question
            ↓
        Retrieval
            ↓
        Relevance check
            ↓
        Prompt construction
            ↓
        Gemini generation
            ↓
        Result
    """

    question = question.strip()

    if not question:

        return {
            "question": question,
            "answer": NO_ANSWER,
            "sources": [],
            "model": None,
            "status": "INVALID_QUESTION",
        }

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    retrieved_chunks = retriever.search(
        question,
        top_k=top_k,
        threshold=threshold,
    )

    print("\nRetrieved evidence:")

    if not retrieved_chunks:

        print(
            "No relevant sources found."
        )

        return {
            "question": question,
            "answer": NO_ANSWER,
            "sources": [],
            "model": None,
            "status": "NO_RELEVANT_SOURCES",
        }

    for chunk in retrieved_chunks:

        print(
            f"- {chunk['source']} "
            f"(score={chunk['score']:.4f})"
        )

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        retrieved_chunks,
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    answer, model_name = generate_with_fallback(
        prompt
    )

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks,
        "model": model_name,
        "status": "SUCCESS",
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    result = generate_answer(
        question
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        result["answer"]
    )

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    print(
        result["model"]
    )

    print("\n" + "=" * 70)
    print("STATUS")
    print("=" * 70)

    print(
        result["status"]
    )

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    if result["sources"]:

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"(chunk={source['chunk_id']}, "
                f"score={source['score']:.4f})"
            )

    else:

        print(
            "No relevant sources found."
        )