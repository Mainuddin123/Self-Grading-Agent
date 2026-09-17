from agent import run_agent


# ============================================================
# SELECT ONE TEST
# ============================================================
# 0 = all
# 1..24 = individual test

TEST_CASE = 24

# ============================================================
# EXPECTED TEST CASES
# ============================================================

TESTS = [

    # 1
    {
        "question": "How many paid annual leave days do full-time employees receive?",
        "keywords": ["18"],
        "expected_status": "SUCCESS",
    },

    # 2
    {
        "question": "What is the annual learning budget?",
        "keywords": ["20,000", "20000", "INR"],
        "expected_status": "SUCCESS",
    },

    # 3
    {
        "question": "What is the company's remote work policy?",
        "keywords": ["3 days", "three days"],
        "expected_status": "SUCCESS",
    },

    # 4
    {
        "question": "Can an employee carry unused annual leave into the next year?",
        "keywords": ["5 days", "carried forward"],
        "expected_status": "SUCCESS",
    },

    # 5
    {
        "question": "What is the company's dental insurance coverage?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },

    # 6
    {
        "question": "Who is the president of India?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },

    # 7
    {
        "question": "What is the company's employee stock trading policy?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },

    # 8
    {
        "question": "What is the company's equipment replacement policy?",
        "keywords": [],
        "expected_status": "NO_SUPPORTED_ANSWER",
    },

    # 9
    {
        "question": "If an employee uses 13 days of annual leave, how many days remain?",
        "keywords": ["5"],
        "expected_status": "SUCCESS",
    },

    # 10
    {
        "question": "If an employee uses 7 days of annual leave, how many days remain?",
        "keywords": ["11"],
        "expected_status": "SUCCESS",
    },

    # 11
    {
        "question": "If an employee uses 5 days of annual leave, how many days remain?",
        "keywords": ["13"],
        "expected_status": "SUCCESS",
    },

    # 12
    {
        "question": "If an employee has 18 days of annual leave and carries forward 5 unused days from last year, how many days are available in total?",
        "keywords": ["23"],
        "expected_status": "SUCCESS",
    },

    # 13
    {
        "question": "If an employee has 18 days and carries forward 5 unused days, how many are available?",
        "keywords": ["23"],
        "expected_status": "SUCCESS",
    },

    # 14
    {
        "question": "How many annual leave days remain if an employee uses 7 days?",
        "keywords": ["11"],
        "expected_status": "SUCCESS",
    },

    # 15
    {
        "question": "Can an employee carry forward 3 unused annual leave days?",
        "keywords": ["5 days", "carried forward"],
        "expected_status": "SUCCESS",
    },

    # 16
    {
        "question": "Can unused annual leave be carried forward?",
        "keywords": ["5 days", "carried forward"],
        "expected_status": "SUCCESS",
    },

    # 17
    {
        "question": "How much parental leave is provided?",
        "keywords": ["12 weeks"],
        "expected_status": "SUCCESS",
    },

    # 18
    {
        "question": "What is the parental leave policy?",
        "keywords": ["12 weeks", "paid leave", "parental leave"],
        "expected_status": "SUCCESS",
    },

    # 19
    {
        "question": "What is the equipment return policy?",
        "keywords": ["return", "equipment"],
        "expected_status": "SUCCESS",
    },

    # 20
    {
        "question": "Do employees have to return company equipment when employment ends?",
        "keywords": ["return", "equipment"],
        "expected_status": "SUCCESS",
    },

    # 21
    {
        "question": "What is the company's equipment replacement policy?",
        "keywords": [],
        "expected_status": "NO_SUPPORTED_ANSWER",
    },

    # 22
    {
        "question": "Who is the president of India?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },

    # 23
    {
        "question": "What is the company's dental insurance policy?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },

    # 24
    {
        "question": "What is the company's stock trading policy?",
        "keywords": [],
        "expected_status": "NO_RELEVANT_SOURCES",
    },
]


# ============================================================
# VALIDATION
# ============================================================

FALLBACK = (
    "I could not find this information in the provided policies."
)


def validate(test, result):

    expected = test["expected_status"]
    actual = result.get("status")

    print("\nEXPECTED STATUS :", expected)
    print("ACTUAL STATUS   :", actual)

    # --------------------------------------------------------
    # STATUS CHECK
    # --------------------------------------------------------

    if actual != expected:
        print("STATUS CHECK    : FAIL")
        return False

    answer = result.get("answer", "")

    # --------------------------------------------------------
    # NO RELEVANT SOURCES
    # --------------------------------------------------------

    if expected == "NO_RELEVANT_SOURCES":

        if answer.strip() != FALLBACK:
            print("ANSWER CHECK    : FAIL")
            return False

        if result.get("final_score", 0.0) != 0.0:
            print("SCORE CHECK     : FAIL")
            return False

        if result.get("attempts", 0) != 0:
            print("ATTEMPT CHECK   : FAIL")
            return False

        print("VALIDATION      : PASS")
        return True

    # --------------------------------------------------------
    # NO SUPPORTED ANSWER
    # --------------------------------------------------------

    if expected == "NO_SUPPORTED_ANSWER":

        if answer.strip() != FALLBACK:
            print("ANSWER CHECK    : FAIL")
            return False

        if result.get("final_score", 0.0) != 0.0:
            print("SCORE CHECK     : FAIL")
            return False

        if result.get("confidence") != "I don't know":
            print("CONFIDENCE CHECK: FAIL")
            return False

        if result.get("attempts", 0) < 1:
            print("ATTEMPT CHECK   : FAIL")
            return False

        print("VALIDATION      : PASS")
        return True

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if expected == "SUCCESS":

        keywords = test.get("keywords", [])

        if keywords:

            answer_lower = answer.lower()

            matched = any(
                keyword.lower() in answer_lower
                for keyword in keywords
            )

            if not matched:
                print("KEYWORD CHECK   : FAIL")
                print("ANSWER          :", answer)
                return False

        score = float(
            result.get("final_score", 0.0)
        )

        if score < 0.70:
            print("SCORE CHECK     : FAIL")
            print("SCORE           :", score)
            return False

        if result.get("attempts", 0) < 1:
            print("ATTEMPT CHECK   : FAIL")
            return False

        print("VALIDATION      : PASS")
        return True

    return False


# ============================================================
# RUN ONE TEST
# ============================================================

def run_one(index):

    test = TESTS[index - 1]
    question = test["question"]

    print("\n")
    print("=" * 75)
    print(f"TEST {index}/24")
    print("=" * 75)

    print("\nQUESTION:")
    print(question)

    try:

        result = run_agent(
            question,
            top_k=3,
            threshold=0.50,
        )

        print("\nANSWER:")
        print(result.get("answer"))

        print("\nCONFIDENCE:")
        print(result.get("confidence"))

        print("\nSCORE:")
        print(result.get("final_score"))

        print("\nATTEMPTS:")
        print(result.get("attempts"))

        print("\nSOURCES:")

        for source in result.get("sources", []):
            print(source)

        passed = validate(
            test,
            result
        )

        print("\nFINAL TEST RESULT:")

        if passed:
            print("PASS")
        else:
            print("FAIL")

        return passed

    except Exception as error:

        print("\nTEST ERROR:")
        print(error)

        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("SELF-GRADING RAG AGENT — INDIVIDUAL END-TO-END TEST")
    print("=" * 75)

    print(f"\nSelected TEST_CASE: {TEST_CASE}")

    # --------------------------------------------------------
    # RUN ALL TESTS
    # --------------------------------------------------------

    if TEST_CASE == 0:

        passed = 0
        failed = 0

        for i in range(
            1,
            len(TESTS) + 1
        ):

            if run_one(i):
                passed += 1
            else:
                failed += 1

        print("\n")
        print("=" * 75)
        print("FINAL SUMMARY")
        print("=" * 75)

        print("TOTAL  :", len(TESTS))
        print("PASSED :", passed)
        print("FAILED :", failed)

    # --------------------------------------------------------
    # RUN INDIVIDUAL TEST
    # --------------------------------------------------------

    elif 1 <= TEST_CASE <= len(TESTS):

        run_one(TEST_CASE)

    # --------------------------------------------------------
    # INVALID TEST NUMBER
    # --------------------------------------------------------

    else:

        print(
            f"Invalid TEST_CASE. "
            f"Choose 0-{len(TESTS)}."
        )