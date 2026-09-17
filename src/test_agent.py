"""
Out-of-benchmark tests for the Self-Grading RAG Agent.

These tests check:
1. Normal answerable questions
2. Multi-detail questions
3. Unsupported questions
4. Unrelated questions
5. Hallucination resistance
6. Abstention behavior
"""

import subprocess
import sys


def run_agent(question: str) -> str:
    """Run agent.py and return terminal output."""

    process = subprocess.run(
        [sys.executable, "src/agent.py"],
        input=question + "\nexit\n",
        text=True,
        capture_output=True,
        timeout=120,
    )

    return process.stdout + process.stderr


def test_answerable_question():
    output = run_agent(
        "How many paid annual leave days do full-time employees receive?"
    )

    assert "18 days" in output


def test_learning_budget():
    output = run_agent(
        "What is the annual learning budget?"
    )

    assert "20,000" in output


def test_remote_work():
    output = run_agent(
        "What is the company remote work policy?"
    )

    assert "remote" in output.lower()


def test_unsupported_equipment_replacement():
    output = run_agent(
        "What is the company's equipment replacement policy?"
    )

    assert "NO_SUPPORTED_ANSWER" in output


def test_unrelated_question():
    output = run_agent(
        "Who is the president of India?"
    )

    assert (
        "No relevant sources found" in output
        or "NO_RELEVANT_SOURCES" in output
    )


def test_hallucination_resistance():
    output = run_agent(
        "What is the company's policy for unlimited paid vacation?"
    )

    assert (
        "NO_SUPPORTED_ANSWER" in output
        or "No relevant sources found" in output
        or "not supported" in output.lower()
    )