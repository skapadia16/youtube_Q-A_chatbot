# 🎥 YouTube RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that allows users to ask questions about a YouTube video's transcript.

The application fetches the YouTube transcript, splits it into smaller chunks, converts those chunks into embeddings, stores them in a FAISS vector database, retrieves the most relevant information for a user's question, and generates an answer using **GPT-OSS-120B** through Hugging Face.

---

## 🚀 Features

* 🎥 Fetch transcript from YouTube videos
* ✂️ Split transcript into meaningful chunks
* 🧠 Generate text embeddings using Hugging Face
* 🔎 Semantic similarity search using FAISS
* 🤖 Generate answers using GPT-OSS-120B
* 🔗 Built using LangChain Runnable Chains
* 💬 Ask questions directly about the video
* 🔐 Hugging Face API token through environment variables

---

## 🏗️ Architecture

```text
                 YouTube Video
                       │
                       ▼
              YouTube Transcript
                       │
                       ▼
                Text Splitting
                       │
                       ▼
              Hugging Face Embeddings
                       │
                       ▼
                 FAISS Vector DB
                       │
                       │
User Question ─────────┤
                       ▼
                  Retriever
                       │
                       ▼
                Relevant Chunks
                       │
                       ▼
              PromptTemplate
                       │
                       ▼
             GPT-OSS-120B (HF)
                       │
                       ▼
                  Final Answer
```

---

## 🛠️ Technologies Used

| Technology             | Purpose                           |
| ---------------------- | --------------------------------- |
| Python                 | Main programming language         |
| LangChain              | RAG pipeline and chain management |
| YouTube Transcript API | Fetch YouTube transcripts         |
| Hugging Face           | Embeddings and LLM                |
| FAISS                  | Vector database                   |
| Sentence Transformers  | Text embeddings                   |
| GPT-OSS-120B           | Answer generation                 |

---

## 📂 Project Structure

```text
YT_chat/
│
├── venv/
│
├── main.py
│
├── requirements.txt
│
├── .env
│
├── .gitignore
│
└── README.md
```

> `venv/` and `.env` should not be uploaded to GitHub.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project

```bash
cd YT_chat
```

### 3. Create virtual environment

```bash
python -m venv venv
```

### 4. Activate virtual environment

### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt` yet:

```bash
pip install youtube-transcript-api
pip install langchain
pip install langchain-text-splitters
pip install langchain-huggingface
pip install langchain-community
pip install faiss-cpu
pip install sentence-transformers
pip install huggingface-hub
```

Then:

```bash
pip freeze > requirements.txt
```

---

## 🔑 Hugging Face API Token

Create a Hugging Face API token and store it in an environment variable.

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
```

Then load the environment variable in Python.

```python
import os

token = os.getenv("HF_TOKEN")
```

### ⚠️ Important

Never upload your `.env` file or API token to GitHub.

Add this to `.gitignore`:

```gitignore
venv/
.env
__pycache__/
*.pyc
```

---

## ▶️ Run the Project

Run:

```bash
python main.py
```

The program will ask:

```text
Enter question related to video:
```

Example:

```text
Enter question related to video: What is the main topic of this video?
```

The chatbot retrieves the relevant transcript sections and generates an answer.

---

## 🔗 LangChain RAG Chain

The main RAG pipeline is implemented using LangChain Runnable components:

```python
rag_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
)
```

The chain performs:

```text
Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
Context Formatting
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

---

## 🧠 How RAG Works

### 1. Retrieval

The YouTube transcript is divided into smaller chunks.

```python
chunks = splitter.create_documents(
    [transcript_text]
)
```

These chunks are converted into vector embeddings.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The embeddings are stored in FAISS:

```python
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
```

---

### 2. Augmentation

When the user asks a question, the retriever searches for the most relevant transcript chunks.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

The retrieved chunks are added to the prompt as context.

---

### 3. Generation

The context and question are sent to GPT-OSS-120B through Hugging Face.

```python
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
```

The generated answer is returned to the user.

---

## 📌 Example

### User Question

```text
What is the main concept explained in the video?
```

### RAG Process

```text
User Question
      ↓
FAISS Similarity Search
      ↓
Top 4 Relevant Chunks
      ↓
Context + Question
      ↓
GPT-OSS-120B
      ↓
Generated Answer
```

---

## 🎯 Why RAG?

A normal LLM may not have access to the specific content of a YouTube video.

RAG solves this by:

1. Retrieving relevant information from the video's transcript.
2. Providing that information to the LLM.
3. Generating an answer based on the retrieved context.

This helps the chatbot answer questions specifically about the selected video.

---

## 🔮 Future Improvements

* [ ] Add Streamlit web interface
* [ ] Support multiple YouTube videos
* [ ] Add conversation memory
* [ ] Add chat history
* [ ] Support videos with multiple languages
* [ ] Add source/reference chunks to answers
* [ ] Add PDF document support
* [ ] Deploy the chatbot online
* [ ] Add voice-based questions
* [ ] Add video summarization

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

* Python
* LangChain
* Retrieval-Augmented Generation
* Vector databases
* FAISS
* Embeddings
* Semantic search
* Prompt engineering
* Hugging Face APIs
* LLM integration
* LangChain Runnable architecture

---

## 👨‍💻 Author

**Shreyas Kapadiya**

AI Engineer & Data Science Enthusiast

---

## ⭐ If you find this project useful

Give the repository a ⭐ and feel free to explore, modify, and improve the project.
