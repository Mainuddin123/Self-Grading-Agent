"""
Calibration utilities for the Self-Grading RAG Agent.

This module converts textual confidence levels into probabilities
and measures how well confidence matches actual correctness.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CalibrationRecord:
    """One evaluated answer."""

    confidence: float
    correct: bool


def confidence_to_probability(confidence: str) -> float:
    """
    Convert the agent's textual confidence into a numeric probability.
    """

    value = confidence.strip().lower()

    mapping = {
        "high confidence": 0.90,
        "medium confidence": 0.65,
        "low confidence": 0.35,
        "i don't know": 0.05,
        "don't know": 0.05,
        "unknown": 0.05,
    }

    return mapping.get(value, 0.50)


def calibration_summary(
    records: Iterable[CalibrationRecord],
) -> dict:
    """
    Calculate basic calibration statistics.

    Calibration error is the absolute difference between:
        average predicted confidence
        and
        actual accuracy
    """

    records = list(records)

    if not records:
        return {
            "count": 0,
            "average_confidence": 0.0,
            "accuracy": 0.0,
            "calibration_error": 0.0,
        }

    average_confidence = sum(
        record.confidence for record in records
    ) / len(records)

    accuracy = sum(
        1 for record in records if record.correct
    ) / len(records)

    calibration_error = abs(
        average_confidence - accuracy
    )

    return {
        "count": len(records),
        "average_confidence": round(average_confidence, 4),
        "accuracy": round(accuracy, 4),
        "calibration_error": round(calibration_error, 4),
    }


def expected_calibration_error(
    records: Iterable[CalibrationRecord],
) -> float:
    """
    Calculate a simple Expected Calibration Error (ECE).

    Each confidence level is treated as a probability prediction.
    """

    records = list(records)

    if not records:
        return 0.0

    total_error = 0.0

    for record in records:
        total_error += abs(
            record.confidence - float(record.correct)
        )

    return round(total_error / len(records), 4)


def confidence_statistics(
    records: Iterable[CalibrationRecord],
) -> dict:
    """
    Return correctness statistics grouped by confidence level.
    """

    records = list(records)

    groups = {
        "high": [],
        "medium": [],
        "low": [],
        "unknown": [],
    }

    for record in records:
        if record.confidence >= 0.80:
            groups["high"].append(record.correct)
        elif record.confidence >= 0.50:
            groups["medium"].append(record.correct)
        elif record.confidence > 0.10:
            groups["low"].append(record.correct)
        else:
            groups["unknown"].append(record.correct)

    result = {}

    for name, values in groups.items():
        count = len(values)
        correct = sum(values)

        result[name] = {
            "count": count,
            "correct": correct,
            "accuracy": round(correct / count, 4)
            if count
            else 0.0,
        }

    return result


if __name__ == "__main__":
    # Small standalone demonstration.
    records = [
        CalibrationRecord(0.90, True),
        CalibrationRecord(0.90, True),
        CalibrationRecord(0.65, True),
        CalibrationRecord(0.35, False),
    ]

    print("=" * 60)
    print("CALIBRATION CHECK")
    print("=" * 60)

    summary = calibration_summary(records)

    print(f"Samples             : {summary['count']}")
    print(
        f"Average confidence  : "
        f"{summary['average_confidence']:.2f}"
    )
    print(
        f"Accuracy            : "
        f"{summary['accuracy']:.2f}"
    )
    print(
        f"Calibration error   : "
        f"{summary['calibration_error']:.2f}"
    )
    print(
        f"ECE                 : "
        f"{expected_calibration_error(records):.2f}"
    )

    print("\nConfidence statistics:")

    for level, stats in confidence_statistics(records).items():
        print(
            f"{level:8} | "
            f"count={stats['count']} | "
            f"correct={stats['correct']} | "
            f"accuracy={stats['accuracy']:.2f}"
        )

        