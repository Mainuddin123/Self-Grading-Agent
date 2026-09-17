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
        "GOOGLE_API_KEY not found. "
        "Add it to the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# MODEL FALLBACK ORDER
# ============================================================

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


# ============================================================
# GENERATION WITH FALLBACK
# ============================================================

def generate_with_fallback(prompt):
    """
    Generate a response using Gemini.

    Models are attempted in order.
    If a model fails because of quota,
    rate limits, or another API error,
    the next model is attempted.
    """

    last_error = None

    for model in MODELS:

        print()
        print(f"Trying model: {model}")

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            if not response.text:
                raise ValueError(
                    "Model returned an empty response."
                )

            print(
                f"Model selected: {model}"
            )

            return {
                "text": response.text.strip(),
                "model": model,
                "status": "SUCCESS",
                "error": None,
            }

        except Exception as error:

            last_error = error

            print(
                f"Model {model} failed: {error}"
            )

            print(
                "Trying next available model..."
            )

    print()
    print("All Gemini models failed.")

    return {
        "text": None,
        "model": None,
        "status": "FAILED",
        "error": str(last_error),
    }