# Self-Grading RAG Agent



A retrieval-augmented generation (RAG) agent that answers questions from a controlled set of company policy documents and automatically evaluates its own answers for correctness, relevance, and grounding.



\## Overview



The Self-Grading RAG Agent is designed to reduce unsupported or hallucinated answers by combining:



\- Document retrieval

\- Evidence-based answer generation

\- Evidence support checking

\- Automatic self-grading

\- Confidence estimation

\- Model fallback handling

\- Evaluation across multiple question categories

\- Calibration analysis

\- Automated tests



The system is intentionally designed to return a controlled `NO\_SUPPORTED\_ANSWER` response when the retrieved evidence does not directly support the requested information.



\## Architecture



```text

User Question

&#x20;     |

&#x20;     v

Question Processing

&#x20;     |

&#x20;     v

Document Retrieval

&#x20;     |

&#x20;     v

Relevant Evidence

&#x20;     |

&#x20;     +----------------------+

&#x20;     |                      |

&#x20;     v                      v

Evidence Support Check    No Relevant Evidence

&#x20;     |                      |

&#x20;     v                      v

Answer Generation       NO\_SUPPORTED\_ANSWER

&#x20;     |

&#x20;     v

Self-Grading

&#x20;     |

&#x20;     +-----------------------------+

&#x20;     |             |               |

&#x20;     v             v               v

Correctness     Relevance       Grounding

&#x20;     |             |               |

&#x20;     +-------------+---------------+

&#x20;                   |

&#x20;                   v

&#x20;            Overall Score

&#x20;                   |

&#x20;                   v

&#x20;             Confidence

&#x20;                   |

&#x20;                   v

&#x20;            Final Result





\# Key Features



1\. RAG-based Question Answering



The agent retrieves relevant chunks from the company policy documents before generating an answer.



2\. Evidence-Grounded Responses



The generated answer is checked against the retrieved evidence instead of relying only on the language model.



3\. Hallucination Resistance



Questions outside the available knowledge base can result in:

&#x20;               NO\_RELEVANT\_SOURCES

&#x20;                     or 

&#x20;               NO\_SUPPORTED\_ANSWER



This prevents the system from inventing information that is not supported by the provided documents.



4\. Self-Grading



Supported answers are evaluated using:



Correctness

Relevance

Grounding

Overall score

5\. Confidence Estimation



The system reports confidence levels such as:



High confidence

Medium confidence

Low confidence

I don't know





6\. Model Fallback



If a configured Gemini model reaches a quota/rate limit, the system attempts another available grading/generation model.





7\. Evaluation Framework



The project contains an automated evaluation dataset covering:



ANSWERABLE

UNANSWERABLE

PARTIALLY\_SUPPORTED

TRAP



8\. Calibration Evaluation



The project evaluates whether reported confidence corresponds with observed correctness.



Calibration metrics include:



Average confidence

Accuracy

Calibration error

Expected Calibration Error (ECE)

Confidence breakdown



\# Project Structure



Self-Grading-Agent/

â”‚

â”œâ”€â”€ data/

â”‚   â”œâ”€â”€ docs/

â”‚   â”‚   â”œâ”€â”€ 01\_remote\_work\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 02\_leave\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 03\_expense\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 04\_learning\_budget.txt

â”‚   â”‚   â”œâ”€â”€ 05\_security\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 06\_travel\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 07\_performance\_review.txt

â”‚   â”‚   â”œâ”€â”€ 08\_equipment\_policy.txt

â”‚   â”‚   â”œâ”€â”€ 09\_recruitment\_referral.txt

â”‚   â”‚   â”œâ”€â”€ 10\_data\_retention.txt

â”‚   â”‚   â”œâ”€â”€ 11\_code\_of\_conduct.txt

â”‚   â”‚   â””â”€â”€ 12\_parental\_benefits.txt

â”‚   â”‚

â”‚   â”œâ”€â”€ evaluation.json

â”‚   â””â”€â”€ evaluation\_results.json

â”‚

â”œâ”€â”€ reports/

â”‚   â””â”€â”€ calibration\_results.json

â”‚

â”œâ”€â”€ src/

â”‚   â”œâ”€â”€ agent.py

â”‚   â”œâ”€â”€ calibration.py

â”‚   â”œâ”€â”€ evaluate.py

â”‚   â”œâ”€â”€ grader.py

â”‚   â”œâ”€â”€ llm.py

â”‚   â”œâ”€â”€ rag.py

â”‚   â”œâ”€â”€ retriever.py

â”‚   â”œâ”€â”€ run\_calibration.py

â”‚   â”œâ”€â”€ test\_agent.py

â”‚   â””â”€â”€ test\_retriever.py

â”‚

â”œâ”€â”€ .gitignore

â”œâ”€â”€ evaluation\_results.json

â”œâ”€â”€ requirements.txt

â””â”€â”€ README.md





\# Technologies Used:



Python

Google Gemini

Google Generative AI

Sentence Transformers

FAISS

NumPy

Pandas

Matplotlib

Pytest

python-dotenv

Installation



1\. Clone the repository:



git clone https://github.com/Mainuddin123/Self-Grading-Agent.git

cd Self-Grading-Agent



2\. Create a virtual environment:



python -m venv .venv



3\. Activate the environment on Windows:



.venv\\Scripts\\Activate.ps1



4\. Install dependencies:



pip install -r requirements.txt





\# Environment Configuration



Create a .env file in the project root and configure the required Google API credentials used by the application.



Example: GOOGLE\_API\_KEY = your_api_key_here



Do not commit API keys or other secrets to GitHub.





\# Running the Agent



From the project root: python src/agent.py

The application accepts questions interactively.



Example:

Enter your question (or type 'exit'):

How many paid annual leave days do full-time employees receive?



The system retrieves supporting documents, generates an answer, evaluates it, and displays the final result.





\# Example Supported Question

How many paid annual leave days do full-time employees receive?



Example result:



ANSWER:

Full-time employees receive 18 days of paid annual leave per calendar year.



CONFIDENCE: High confidence



FINAL SCORE: 1.00

STATUS: SUCCESS



\# Unsupported Question Handling



For information that is not supported by the provided documents, the system can return:

NO\_RELEVANT\_SOURCES



For retrieved evidence that does not directly support the requested concept, the system can return:

NO\_SUPPORTED\_ANSWER



Example:



Question:

What is the company's equipment replacement policy?



Evidence support check: NOT SUPPORTED



Returning NO\_SUPPORTED\_ANSWER.



This behaviour is an important part of the project's hallucination-resistance design.





\# Automated Evaluation



Run the complete evaluation with: python src/evaluate.py



The evaluation categorises questions into:



ANSWERABLE

UNANSWERABLE

PARTIALLY\_SUPPORTED

TRAP



The completed evaluation currently contains 24 questions.



Latest evaluation result:



Completed questions: 24

Passed questions: 24

Failed questions    : 0

Pass rate           : 100.00%

Average attempts: 1.00



\# Category results:

ANSWERABLE          10/10 passed

UNANSWERABLE         5/5 passed

PARTIALLY\_SUPPORTED  5/5 passed

TRAP                 4/4 passed



\# Automated Tests



Run the test suite with: python -m pytest -v

Current test suite:  6 passed



\# Tests cover:



Answerable questions

Learning budget retrieval

Remote work retrieval

Unsupported equipment replacement handling

Unrelated questions

Hallucination resistance



\# Calibration Evaluation



Run: python src/run\_calibration.py



The calibration evaluation currently processes 24 samples.



Latest recorded calibration results:

Total samples: 24

Average confidence: 0.7229

Accuracy            : 0.7917

Calibration error: 0.0687

ECE                 : 0.0896



Results are saved to: reports/calibration\_results.json



**# Evaluation Output**



**The project stores evaluation results in:** evaluation\_results.json



and calibration results in: reports/calibration\_results.json

These files provide reproducible evidence of the evaluation performed on the agent.





\# Safety and Reliability Design



The system uses several mechanisms to improve reliability:



Retrieve evidence before generation.

Check whether evidence supports the requested concept.

Avoid answering when evidence is insufficient.

Return controlled unsupported-answer states.

Grade generated answers against retrieved evidence.

Report confidence alongside the result.

Evaluate the system using answerable and adversarial/trap questions.

Measure confidence calibration separately.





\# Limitations



The agent's knowledge is limited to the supplied policy documents.

Answers outside the document collection may intentionally return no supported answer.

Model availability can depend on API quota and rate limits.

Confidence calibration depends on the evaluation dataset and should not be interpreted as a universal measure of real-world reliability.

The system does not provide general web-based knowledge retrieval.





\# Future Improvements



Potential extensions include:



Web-based retrieval with controlled source validation

More advanced reranking

Hybrid keyword + vector retrieval

Query rewriting

Better confidence calibration

Larger evaluation datasets

LLM-as-a-judge evaluation comparison

Retrieval evaluation metrics such as Recall@K and MRR

RAG evaluation frameworks

Web/API interface

Persistent evaluation dashboards.



Author



Shaik Khaja Mainuddin



GitHub:

https://github.com/Mainuddin123



LinkedIn:

https://www.linkedin.com/in/khaja-mainuddin-sk-958436348



Repository



https://github.com/Mainuddin123/Self-Grading-Agent

'@ | Set-Content README.md -Encoding UTF8



\### 2. Verify it



Run:



```powershell

Get-Content README.md





































