import json
from pathlib import Path

from calibration import (
    CalibrationRecord,
    calibration_summary,
    expected_calibration_error,
    confidence_statistics,
    confidence_to_probability,
)


RESULTS_FILE = Path("evaluation_results.json")


def load_results():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{RESULTS_FILE} was not found."
        )

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("results", "evaluations", "questions"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Could not find evaluation results in JSON file."
    )


def main():
    results = load_results()

    records = []

    for item in results:
        confidence = str(
            item.get("confidence", "I don't know")
        )

        probability = confidence_to_probability(
            confidence
        )

        # IMPORTANT:
        # Use the evaluator's actual pass/fail decision.
        # This correctly handles "I don't know" when abstention
        # is the correct decision.
        correct = bool(item.get("passed", False))

        records.append(
            CalibrationRecord(
                confidence=probability,
                correct=correct,
            )
        )

    summary = calibration_summary(records)
    ece = expected_calibration_error(records)
    statistics = confidence_statistics(records)

    # Count calibration-specific failure modes.
    over_cautious_abstentions = 0
    dangerous_overconfidence = 0

    for item in results:
        confidence = str(
            item.get("confidence", "I don't know")
        ).strip().lower()

        passed = bool(item.get("passed", False))
        category = str(
            item.get("category", "")
        ).strip().upper()

        # An answerable question answered with "I don't know"
        # is an over-cautious abstention.
        if (
            confidence in {
                "i don't know",
                "don't know",
                "unknown",
            }
            and category in {
                "ANSWERABLE",
                "PARTIALLY_SUPPORTED",
            }
            and not passed
        ):
            over_cautious_abstentions += 1

        # A failed answer given high confidence is dangerous
        # overconfidence.
        if (
            confidence == "high confidence"
            and not passed
        ):
            dangerous_overconfidence += 1

    print()
    print("=" * 70)
    print("REAL CALIBRATION EVALUATION")
    print("=" * 70)

    print(f"Total samples              : {summary['count']}")
    print(
        f"Average confidence         : "
        f"{summary['average_confidence']:.2f}"
    )
    print(
        f"Decision accuracy          : "
        f"{summary['accuracy']:.2f}"
    )
    print(
        f"Calibration error           : "
        f"{summary['calibration_error']:.2f}"
    )
    print(f"ECE                         : {ece:.2f}")
    print(
        f"Over-cautious abstentions   : "
        f"{over_cautious_abstentions}"
    )
    print(
        f"Dangerous overconfidence    : "
        f"{dangerous_overconfidence}"
    )

    print()
    print("CONFIDENCE BREAKDOWN")
    print("-" * 70)

    for level, stats in statistics.items():
        print(
            f"{level:8} | "
            f"count={stats['count']:2d} | "
            f"correct={stats['correct']:2d} | "
            f"accuracy={stats['accuracy']:.2f}"
        )

    output = {
        "total_samples": summary["count"],
        "average_confidence": summary["average_confidence"],
        "accuracy": summary["accuracy"],
        "calibration_error": summary["calibration_error"],
        "ece": ece,
        "over_cautious_abstentions": over_cautious_abstentions,
        "dangerous_overconfidence": dangerous_overconfidence,
        "confidence_breakdown": statistics,
    }

    output_file = Path(
        "reports/calibration_results.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("Calibration results saved to:")
    print(output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
    