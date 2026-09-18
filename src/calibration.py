"""
Calibration utilities for the Self-Grading RAG Agent.

The project uses three confidence outcomes:
- High confidence
- Low confidence
- I don't know
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CalibrationRecord:
    """One evaluated answer."""

    confidence: float
    correct: bool


def confidence_to_probability(confidence: str) -> float:
    """Convert textual confidence into a numeric probability."""

    value = confidence.strip().lower()

    mapping = {
        "high confidence": 0.90,
        "low confidence": 0.35,
        "i don't know": 0.05,
        "don't know": 0.05,
        "unknown": 0.05,
    }

    return mapping.get(value, 0.50)


def calibration_summary(
    records: Iterable[CalibrationRecord],
) -> dict:
    """Calculate calibration statistics."""

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
        "average_confidence": round(
            average_confidence, 4
        ),
        "accuracy": round(
            accuracy, 4
        ),
        "calibration_error": round(
            calibration_error, 4
        ),
    }


def expected_calibration_error(
    records: Iterable[CalibrationRecord],
) -> float:
    """
    Calculate the project's simple ECE-style metric.

    This implementation averages the absolute difference
    between predicted confidence probability and correctness.
    """

    records = list(records)

    if not records:
        return 0.0

    total_error = sum(
        abs(
            record.confidence
            - float(record.correct)
        )
        for record in records
    )

    return round(
        total_error / len(records),
        4
    )


def confidence_statistics(
    records: Iterable[CalibrationRecord],
) -> dict:
    """Group correctness into the three project confidence levels."""

    records = list(records)

    groups = {
        "high": [],
        "low": [],
        "unknown": [],
    }

    for record in records:

        if record.confidence >= 0.80:
            groups["high"].append(record.correct)

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
            "accuracy": round(
                correct / count,
                4
            ) if count else 0.0,
        }

    return result
