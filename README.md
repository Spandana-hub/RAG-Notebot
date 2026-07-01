# 📚 NoteBot — RAG-Powered PDF Study Assistant

> **Upload your notes. Ask anything. Get structured, intelligent answers powered by Google Gemini.**

NoteBot is a Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Google Gemini**. Upload any PDF (lecture notes, textbooks, research papers) and ask questions — NoteBot retrieves the most relevant chunks and generates clear, well-structured answers using the Gemini LLM.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-notebot-2xljap9yeez5e34k5pkls2.streamlit.app/)


---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF file directly from the sidebar
- 🔍 **Semantic Search** — Uses FAISS vector store and Google Gemini Embeddings to find the most relevant content
- 🤖 **Gemini-Powered Answers** — Generates structured, detailed answers using the latest available Gemini model
- 🔄 **Auto Model Detection** — Automatically selects the best available embedding and chat model for your API key (priority-based fallback)
- ⚡ **Cached Embeddings** — PDF embeddings are cached per session so re-asking questions is instant
- 🛡️ **Rate-Limit Resilience** — Automatic retry with exponential backoff on API quota errors
- ☁️ **Streamlit Cloud Ready** — Supports both `.env` (local) and `st.secrets` (cloud) for API key management

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | Google Gemini (`gemini-2.5-flash`, `2.0-flash`, `1.5-flash`, `1.5-pro`) |
| **Embeddings** | Google Gemini Embeddings (`gemini-embedding-001`, `text-embedding-004`) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| **RAG Framework** | [LangChain](https://www.langchain.com/) |
| **PDF Parsing** | [PyPDF2](https://pypdf2.readthedocs.io/) |
| **Env Management** | `python-dotenv` |

---

## 📂 Project Structure

```
RAG_Chatbot/
├── code.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Local API key (not committed to Git)
├── .gitignore           # Ignores .env, venv, __pycache__, etc.
└── .streamlit/
    └── config.toml      # Streamlit server configuration
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- A [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/Spandana-hub/RAG-Notebot.git
cd RAG-Notebot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> ⚠️ **Never commit your `.env` file.** It is already listed in `.gitignore`.

### 5. Run the App

```bash
streamlit run code.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your code to a **public GitHub repository** (`.env` is ignored by `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **Advanced Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```
4. Set the **Main file path** to `code.py`
5. Click **Deploy** 🎉

---

## 🧠 How It Works

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF2)
    │
    ▼
Text Chunking (RecursiveCharacterTextSplitter — 1500 chars, 150 overlap)
    │
    ▼
Embedding Generation (Google Gemini Embeddings — batched with retry)
    │
    ▼
FAISS Vector Store (cached per file)
    │
    ▼
User Query ──► Similarity Search (top-6 chunks)
                        │
                        ▼
             LangChain RAG Chain
             (Context + Query → Gemini LLM)
                        │
                        ▼
             Structured Answer displayed in Streamlit
```

---

## 📦 Dependencies

```
streamlit==1.58.0
PyPDF2==3.0.1
langchain==1.3.11
langchain-community==0.4.2
langchain-core==1.4.8
langchain-text-splitters==1.1.2
langchain-google-genai==4.2.5
faiss-cpu==1.14.2
python-dotenv==1.2.2
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♀️ Author

**Spandana**  
Built using LangChain + Google Gemini + Streamlit
