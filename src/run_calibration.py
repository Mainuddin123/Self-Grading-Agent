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

        score = item.get(
            "final_score",
            item.get(
                "overall_score",
                item.get("score", 0.0)
            )
        )

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        probability = confidence_to_probability(
            confidence
        )

        correct = score >= 0.75

        records.append(
            CalibrationRecord(
                confidence=probability,
                correct=correct,
            )
        )

    summary = calibration_summary(records)
    ece = expected_calibration_error(records)
    statistics = confidence_statistics(records)

    print()
    print("=" * 70)
    print("REAL CALIBRATION EVALUATION")
    print("=" * 70)

    print(f"Total samples        : {summary['count']}")
    print(
        f"Average confidence   : "
        f"{summary['average_confidence']:.2f}"
    )
    print(
        f"Accuracy              : "
        f"{summary['accuracy']:.2f}"
    )
    print(
        f"Calibration error     : "
        f"{summary['calibration_error']:.2f}"
    )
    print(f"ECE                   : {ece:.2f}")

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
        "confidence_breakdown": statistics,
    }

    output_file = Path(
        "reports/calibration_results.json"
    )

    output_file.parent.mkdir(exist_ok=True)

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