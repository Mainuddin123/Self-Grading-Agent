# Self-Grading RAG Agent — Architecture

```text
                         ┌───────────────────────┐
                         │      User Question    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Question Processing │
                         │                       │
                         │ • Query preprocessing │
                         │ • Query normalization│
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Document Retrieval  │
                         │                       │
                         │ • Embedding search   │
                         │ • FAISS vector store │
                         │ • Top-K retrieval     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Relevant Evidence    │
                         └───────────┬───────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                       ▼                           ▼
             ┌─────────────────────┐     ┌────────────────────────┐
             │ Evidence Support     │     │ No Relevant Evidence   │
             │ Check                │     │                        │
             └──────────┬──────────┘     └────────────┬───────────┘
                        │                             │
                        │ Supported                   ▼
                        │                   ┌──────────────────────┐
                        │                   │ NO_RELEVANT_SOURCES  │
                        │                   └──────────────────────┘
                        ▼
             ┌─────────────────────┐
             │  Answer Generation  │
             │                     │
             │ • LLM generation    │
             │ • Evidence grounded │
             │ • Source citation   │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │     Self-Grading    │
             │                     │
             │ Evaluate generated  │
             │ answer against      │
             │ retrieved evidence  │
             └──────────┬──────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
 ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
 │  Correctness   │ │  Relevance   │ │   Grounding    │
 │                │ │              │ │                │
 │ Is the answer  │ │ Does the     │ │ Is the answer  │
 │ factually      │ │ answer       │ │ supported by   │
 │ correct?       │ │ address the  │ │ the retrieved  │
 │                │ │ question?    │ │ evidence?      │
 └───────┬────────┘ └──────┬───────┘ └───────┬────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Overall Score    │
                │                     │
                │ Combined evaluation │
                │ score from grading  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Confidence      │
                │                     │
                │ • High confidence   │
                │ • Low confidence    │
                │ • I don't know      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Final Result    │
                │                     │
                │ • SUCCESS           │
                │ • NO_RELEVANT_      │
                │   SOURCES           │
                │ • NO_SUPPORTED_     │
                │   ANSWER            │
                └─────────────────────┘


### Architecture Components:

## 1. User Question

The user submits a natural-language question about the available company policies.

## 2. Question Processing

The question is cleaned and normalized before retrieval to improve semantic search quality.

## 3. Document Retrieval

The system searches the FAISS vector database using embeddings and retrieves the most relevant policy chunks.

## 4. Evidence Support Check

Retrieved evidence is checked to determine whether it actually supports the user's question.

Relevant and supported evidence → Continue to answer generation.
No relevant evidence → Return NO_RELEVANT_SOURCES.
5. Answer Generation

The LLM generates an answer using only the retrieved policy evidence and includes the relevant source information.

## 6. Self-Grading

The generated answer is automatically evaluated across three dimensions:

Correctness — Is the answer factually correct?
Relevance — Does the answer directly address the question?
Grounding — Is the answer supported by the retrieved evidence?
7. Overall Score

The individual evaluation results are combined into an overall score representing the quality of the generated answer.

## 8. Confidence

The system converts the evaluation result into a confidence level such as:

High confidence
Low confidence
I don't know
9. Final Result

The agent returns a structured result containing:

Answer
Status
Confidence
Overall score
Retrieval sources
Number of attempts
Evidence used for evaluation

## Failure Handling:

                    Retrieved Documents
                           │
                           ▼
                  Evidence Support Check
                           │
                ┌──────────┴──────────┐
                │                     │
             Supported            Unsupported
                │                     │
                ▼                     ▼
        Answer Generation     NO_SUPPORTED_ANSWER
                │
                ▼
           Self-Grading

The system is designed to avoid unsupported answers. If retrieved documents exist but do not directly support the requested information, the agent returns:

I could not find this information in the provided policies.

This prevents the LLM from generating an answer based on unsupported information.