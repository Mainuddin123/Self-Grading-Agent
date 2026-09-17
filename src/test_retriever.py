from retriever import Retriever


def main():
    retriever = Retriever()

    questions = [
        "How many paid annual leave days do full-time employees receive?",
        "What is the annual learning budget?",
        "How long are routine system logs retained?",
        "What is the hotel reimbursement limit for domestic travel?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        results = retriever.search(question, top_k=3)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print("Source:", result["source"])
            print("Score:", round(result["score"], 4))
            print("Text preview:")
            print(result["text"][:300])


if __name__ == "__main__":
    main()