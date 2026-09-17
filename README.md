# Self-Grading RAG Agent

A Retrieval-Augmented Generation (RAG) system that answers questions from a controlled set of company policy documents and automatically evaluates the quality of its own answers.

The system is designed to reduce hallucinations by separating:

- Document retrieval
- Evidence relevance checking
- Evidence support validation
- Deterministic policy calculations
- LLM-based answer generation
- Self-grading
- Confidence estimation
- Source attribution
- Fallback handling
- Automated end-to-end testing

---

## Project Overview

Traditional RAG systems retrieve documents and directly send them to an LLM.

This project adds an additional validation layer.

Instead of:

```text
Question
   ↓
Retrieve Documents
   ↓
LLM
   ↓
Answer