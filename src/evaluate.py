import json
from pathlib import Path

from agent import run_agent


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    BASE_DIR / "data" / "evaluation.json"
)

RESULTS_FILE = (
    BASE_DIR / "evaluation_results.json"
)

FALLBACK_ANSWER = (
    "I could not find this information in the provided policies."
)

MINIMUM_SCORE = 0.70


# ============================================================
# LOAD TEST CASES
# ============================================================

def load_test_cases():

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation file not found: "
            f"{EVALUATION_FILE}"
        )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    test_cases = data.get(
        "test_cases",
        []
    )

    if not test_cases:

        raise ValueError(
            "No test cases found in evaluation.json."
        )

    return test_cases


# ============================================================
# KEYWORD CHECK
# ============================================================

def answer_matches_keywords(
    answer,
    keywords,
):

    if not keywords:
        return True

    if not answer:
        return False

    answer_lower = answer.lower()

    return any(
        keyword.lower() in answer_lower
        for keyword in keywords
    )


# ============================================================
# TEST CASE VALIDATION
# ============================================================

def evaluate_result(
    test_case,
    result,
):

    expected_status = test_case.get(
        "expected_status"
    )

    actual_status = result.get(
        "status"
    )

    category = test_case.get(
        "category",
        "UNKNOWN"
    )

    # --------------------------------------------------------
    # STATUS MUST MATCH
    # --------------------------------------------------------

    if actual_status != expected_status:

        return False

    # --------------------------------------------------------
    # NO RELEVANT SOURCES
    # --------------------------------------------------------

    if expected_status == "NO_RELEVANT_SOURCES":

        if (
            result.get("answer", "").strip()
            != FALLBACK_ANSWER
        ):
            return False

        if result.get(
            "final_score",
            0.0
        ) != 0.0:

            return False

        if result.get(
            "attempts",
            0
        ) != 0:

            return False

        return True

    # --------------------------------------------------------
    # NO SUPPORTED ANSWER
    # --------------------------------------------------------

    if expected_status == "NO_SUPPORTED_ANSWER":

        if (
            result.get("answer", "").strip()
            != FALLBACK_ANSWER
        ):
            return False

        if result.get(
            "final_score",
            0.0
        ) != 0.0:

            return False

        if result.get(
            "attempts",
            0
        ) < 1:

            return False

        return True

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if expected_status == "SUCCESS":

        answer = result.get(
            "answer",
            ""
        )

        keywords = test_case.get(
            "expected_keywords",
            []
        )

        # Keyword validation
        if not answer_matches_keywords(
            answer,
            keywords,
        ):
            return False

        # Score validation
        if result.get(
            "final_score",
            0.0
        ) < MINIMUM_SCORE:

            return False

        # At least one attempt
        if result.get(
            "attempts",
            0
        ) < 1:

            return False

        return True

    return False


# ============================================================
# CATEGORY VALIDATION
# ============================================================

def validate_categories(test_cases):

    required_categories = {
        "ANSWERABLE": 10,
        "UNANSWERABLE": 5,
        "PARTIALLY_SUPPORTED": 5,
        "TRAP": 4,
    }

    counts = {}

    for test_case in test_cases:

        category = test_case.get(
            "category",
            "UNKNOWN"
        )

        counts[category] = (
            counts.get(category, 0) + 1
        )

    print("\nCategory distribution:")

    for category, required in (
        required_categories.items()
    ):

        actual = counts.get(
            category,
            0
        )

        status = (
            "OK"
            if actual >= required
            else "INSUFFICIENT"
        )

        print(
            f"{category:22} "
            f"{actual:2} / {required:2} "
            f"{status}"
        )

    return counts


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    test_cases,
    results,
    category_counts,
):

    total_questions = len(
        test_cases
    )

    passed_questions = sum(
        1
        for item in results
        if item["passed"]
    )

    total_score = sum(
        item["score"]
        for item in results
    )

    average_score = (
        total_score / total_questions
        if total_questions
        else 0.0
    )

    pass_rate = (
        passed_questions
        / total_questions
        * 100
        if total_questions
        else 0.0
    )

    attempts = [
        item["attempts"]
        for item in results
        if item["attempts"] > 0
    ]

    average_attempts = (
        sum(attempts) / len(attempts)
        if attempts
        else 0.0
    )

    output = {

        "summary": {
            "total_questions": total_questions,
            "passed_questions": passed_questions,
            "failed_questions": (
                total_questions
                - passed_questions
            ),
            "pass_rate": round(
                pass_rate,
                2
            ),
            "average_score": round(
                average_score,
                4
            ),
            "average_attempts": round(
                average_attempts,
                2
            ),
        },

        "category_distribution":
            category_counts,

        "results":
            results,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():

    print("\n")
    print("=" * 75)
    print(
        "SELF-GRADING RAG AGENT - "
        "24 QUESTION EVALUATION"
    )
    print("=" * 75)

    test_cases = load_test_cases()

    total_questions = len(
        test_cases
    )

    print(
        f"\nLoaded questions: "
        f"{total_questions}"
    )

    category_counts = (
        validate_categories(
            test_cases
        )
    )

    passed_questions = 0
    total_score = 0.0
    total_attempts = 0

    results = []

    # ========================================================
    # RUN EACH TEST
    # ========================================================

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        question = test_case[
            "question"
        ]

        category = test_case.get(
            "category",
            "UNKNOWN"
        )

        print("\n")
        print("=" * 75)
        print(
            f"TEST {index}/{total_questions}"
        )
        print("=" * 75)

        print(
            f"Category : {category}"
        )

        print(
            f"Question : {question}"
        )

        try:

            result = run_agent(
                question,
                top_k=3,
                threshold=0.50,
            )

            passed = evaluate_result(
                test_case,
                result,
            )

            if passed:

                passed_questions += 1

            score = result.get(
                "final_score",
                0.0
            )

            attempts = result.get(
                "attempts",
                0
            )

            total_score += score
            total_attempts += attempts

            result_item = {

                "id":
                    test_case.get(
                        "id",
                        index
                    ),

                "question":
                    question,

                "category":
                    category,

                "expected_status":
                    test_case.get(
                        "expected_status"
                    ),

                "expected_keywords":
                    test_case.get(
                        "expected_keywords",
                        []
                    ),

                "reference_answer":
                    test_case.get(
                        "reference_answer"
                    ),

                "answer":
                    result.get(
                        "answer",
                        ""
                    ),

                "status":
                    result.get(
                        "status"
                    ),

                "confidence":
                    result.get(
                        "confidence"
                    ),

                "score":
                    score,

                "attempts":
                    attempts,

                "sources":
                    result.get(
                        "sources",
                        []
                    ),

                "passed":
                    passed,
            }

            results.append(
                result_item
            )

            print(
                f"\nExpected status : "
                f"{test_case.get('expected_status')}"
            )

            print(
                f"Actual status   : "
                f"{result.get('status')}"
            )

            print(
                f"Score           : "
                f"{score:.2f}"
            )

            print(
                f"Confidence      : "
                f"{result.get('confidence')}"
            )

            print(
                f"Attempts        : "
                f"{attempts}"
            )

            print(
                f"Evaluation      : "
                f"{'PASS' if passed else 'FAIL'}"
            )

        except Exception as error:

            print(
                f"\nERROR: {error}"
            )

            results.append({

                "id":
                    test_case.get(
                        "id",
                        index
                    ),

                "question":
                    question,

                "category":
                    category,

                "expected_status":
                    test_case.get(
                        "expected_status"
                    ),

                "answer":
                    "",

                "status":
                    "ERROR",

                "confidence":
                    None,

                "score":
                    0.0,

                "attempts":
                    0,

                "sources":
                    [],

                "passed":
                    False,

                "error":
                    str(error),
            })

    # ========================================================
    # FINAL METRICS
    # ========================================================

    average_score = (
        total_score
        / total_questions
        if total_questions
        else 0.0
    )

    pass_rate = (
        passed_questions
        / total_questions
        * 100
        if total_questions
        else 0.0
    )

    answered_results = [
        item
        for item in results
        if item["attempts"] > 0
    ]

    average_attempts = (
        sum(
            item["attempts"]
            for item in answered_results
        )
        / len(answered_results)
        if answered_results
        else 0.0
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        test_cases,
        results,
        category_counts,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 75)

    print(
        f"\nCompleted questions : "
        f"{total_questions}"
    )

    print(
        f"Passed questions    : "
        f"{passed_questions}"
    )

    print(
        f"Failed questions    : "
        f"{total_questions - passed_questions}"
    )

    print(
        f"Average score       : "
        f"{average_score:.2f}"
    )

    print(
        f"Pass rate           : "
        f"{pass_rate:.2f}%"
    )

    print(
        f"Average attempts    : "
        f"{average_attempts:.2f}"
    )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )

    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    print("\nCategory summary:")

    for category in [
        "ANSWERABLE",
        "UNANSWERABLE",
        "PARTIALLY_SUPPORTED",
        "TRAP",
    ]:

        category_results = [
            item
            for item in results
            if item["category"] == category
        ]

        category_passed = sum(
            1
            for item in category_results
            if item["passed"]
        )

        print(
            f"{category:22} "
            f"{category_passed}/"
            f"{len(category_results)} passed"
        )

    print("\n")
    print("=" * 75)

    if (
        passed_questions
        == total_questions
    ):

        print(
            "ALL AUTOMATED TESTS PASSED"
        )

    else:

        print(
            "SOME AUTOMATED TESTS FAILED"
        )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_evaluation()
    