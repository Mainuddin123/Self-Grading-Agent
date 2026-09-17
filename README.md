# 🤖 Self-Grading RAG Agent

### Evidence-Grounded Enterprise Policy Assistant

A production-style Retrieval-Augmented Generation (RAG) application that answers questions from enterprise policy documents using retrieved evidence and automatically evaluates whether the generated answer is supported by that evidence.

## 🚀 Live Demo

🔗 **Streamlit App:** https://self-grading-rag-agentgit-pwxgd5jgkdepjelnpzj5de.streamlit.app/

## 📌 Project Overview

The Self-Grading RAG Agent is designed to reduce unsupported or hallucinated answers in document-based question answering.

The system:

- Retrieves relevant policy documents
- Checks whether retrieved evidence supports the question
- Generates an answer using an LLM
- Performs automatic self-grading
- Assigns a confidence level
- Returns a final score
- Distinguishes between:
  - Supported answers
  - No relevant sources
  - Retrieved sources without sufficient support

## 🏗️ Architecture

```text
User Question
      ↓
Query Processing
      ↓
Document Retrieval
      ↓
Relevant Evidence
      ↓
Evidence Support Check
      ↓
LLM Answer Generation
      ↓
Self-Grading
      ↓
Final Score + Confidence + Sources
      ↓
Streamlit UI

<img width="1206" height="1304" alt="ChatGPT Image Sep 18, 2026, 12_37_50 AM" src="https://github.com/user-attachments/assets/4db95104-692a-4e62-ba6f-9afb84311cbd" />

## 🧠 Key Features

### 1. Evidence-Grounded RAG
- Retrieves relevant chunks from enterprise policy documents.
- Generates answers based on retrieved evidence.

### 2. Evidence Support Validation
- Checks whether the retrieved evidence actually supports the user's question.
- Prevents unsupported answers from being generated.

### 3. Self-Grading
Automatically evaluates generated answers using:
- Answer correctness
- Evidence support
- Confidence
- Retrieved sources
- Final score

### 4. Safe Fallbacks
The agent distinguishes between different failure conditions:

- `SUCCESS` — sufficient evidence supports an answer.
- `NO_RELEVANT_SOURCES` — no relevant documents were retrieved.
- `NO_SUPPORTED_ANSWER` — documents were retrieved, but they do not sufficiently support an answer.

### 5. Multi-Model Fallback
- Attempts available Gemini models in sequence.
- Falls back to another model when the preferred model reaches a quota or rate limit.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend / UI:** Streamlit
- **LLM:** Google Gemini
- **Generative AI:** Google Generative AI
- **RAG:** Retrieval-Augmented Generation (RAG)
- **Embeddings:** Sentence Transformers
- **Vector Database:** FAISS
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Testing:** Pytest
- **Environment Management:** Python-dotenv
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

### 6. Transparent Results
The application displays:
- Generated answer
- Confidence level
- Final score
- Retrieved sources
- Retrieval scores
- Evidence support status

## 🔄 RAG Pipeline

### Step 1 — Document Loading
Enterprise policy documents are loaded into the application.

### Step 2 — Document Chunking
Policy documents are split into smaller chunks to improve semantic retrieval.

### Step 3 — Embedding Generation
Each document chunk is converted into a vector representation using sentence-transformer embeddings.

### Step 4 — Vector Retrieval
FAISS performs similarity search to retrieve the most relevant policy chunks for the user's question.

### Step 5 — Evidence Support Check
The retrieved evidence is checked to determine whether it actually supports the question.

### Step 6 — Answer Generation
If sufficient evidence is available, Google Gemini generates an answer grounded in the retrieved policy evidence.

### Step 7 — Self-Grading
The generated answer is automatically evaluated and assigned:
- Final score
- Confidence
- Status
- Retrieved sources

### Step 8 — Final Response
The Streamlit interface presents the answer together with its confidence, score, source information, and evidence status.

## 🧪 Evaluation

The project includes an end-to-end self-grading test suite with **24 test cases** covering:

- Direct policy questions
- Numerical reasoning
- Annual leave calculations
- Leave carry-forward questions
- Parental leave
- Equipment policies
- Unsupported questions
- Questions with no relevant sources

### Example Evaluation

```text
Question:
How many paid annual leave days do full-time employees receive?

Status:
SUCCESS

Answer:
Full-time employees receive 18 days of paid annual leave per calendar year.

Score:
1.00

Confidence:
High confidence

## 🛡️ Grounding & Fallback Behavior

The agent is designed to avoid generating unsupported answers when the provided policy documents do not contain sufficient evidence.

### No Relevant Sources

When no relevant policy documents are retrieved, the agent returns a controlled fallback response instead of generating an answer from unrelated knowledge.

```text
Question:
Who is the president of India?

Status:
NO_RELEVANT_SOURCES

Answer:
I could not find this information in the provided policies.

## 💻 Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Mainuddin123/Self-Grading-RAG-Agent.git
cd Self-Grading-RAG-Agent

2. Create a Virtual Environment : python -m venv.venv

3. Activate the Environment

Windows: .venv\Scripts\activate

4. Install Dependencies: pip install -r requirements.txt

5. Configure API Key
Create a .env file in the project root: ***GOOGLE_API_KEY = your_google_api_key***
Never commit your API key or .env file to GitHub.

6. Run the Application: **streamlit run app.py**
The application will open in your browser.

## 🧪 Testing

The project includes a dedicated end-to-end self-grading test runner for validating the RAG agent.

### Test Runner

```text
src/test_agent.py

The test suite contains 24 test cases covering:

Direct policy questions
Annual leave calculations
Leave carry-forward
Parental leave
Equipment policies
Supported questions
Unsupported questions
Out-of-domain questions
No-relevant-source handling

**Run a Single Test**

To test one specific case, set the TEST_CASE value in src/test_agent.py.

# Example:
TEST_CASE = 18
Then run: python src/test_agent.py
This allows individual test cases to be debugged without running the entire test suite.

# Run All Tests
Set: TEST_CASE = 0
Then run: python src/test_agent.py

## Validation Checks

Each test validates:

Expected status
Actual status
Answer content
Required keywords
Final score
Confidence
Number of attempts
Fallback behavior

### Evaluation Categories:
✅ SUCCESS
Used when the available policy evidence supports the answer.

⚠️ NO_SUPPORTED_ANSWER
Used when documents are retrieved but they do not provide sufficient evidence to answer the question.

ℹ️ NO_RELEVANT_SOURCES
Used when no relevant policy sources are retrieved for the question.

# Example Test:

Question:
If an employee uses 13 days of annual leave, how many days remain?

Expected:
5 days
Expected Status:
SUCCESS

The test runner automatically compares the agent's result with the expected outcome and reports:
VALIDATION: PASS
or:
VALIDATION: FAIL

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Demo

🔗 **Streamlit App:** PASTE_YOUR_LIVE_APP_URL_HERE

### Source Code

🔗 **GitHub Repository:**  
https://github.com/Mainuddin123/Self-Grading-RAG-Agent

---

## 📂 Project Structure

```text
Self-Grading-RAG-Agent/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── policy documents
│
└── src/
    ├── agent.py
    └── test_agent.py

🎯 Project Objective

The objective of this project is to build a reliable enterprise policy assistant using Retrieval-Augmented Generation (RAG).

Unlike a basic RAG application, this system adds evidence validation and self-grading layers to determine whether a generated answer is sufficiently supported by the retrieved policy evidence.

User Question
      ↓
Document Retrieval
      ↓
Evidence Validation
      ↓
Answer Generation
      ↓
Self-Grading
      ↓
Confidence + Score
      ↓
Final Response

The system is designed to provide grounded answers when sufficient evidence is available and controlled fallback responses when the available information does not support an answer.

🔮 Future Improvements:

Display retrieved evidence text directly in the UI
Add dedicated RAG evaluation metrics
Add citation-level verification
Add document upload functionality
Add conversation memory
Add authentication and role-based access
Add production monitoring and logging
Expand the automated evaluation dataset

👨‍💻 Author

Shaik Khaja Mainuddin

B.Tech — Artificial Intelligence & Data Science

GitHub:
https://github.com/Mainuddin123

LinkedIn:
https://www.linkedin.com/in/khaja-mainuddin-sk-958436348

⭐ If you find this project useful, consider starring the repository.
