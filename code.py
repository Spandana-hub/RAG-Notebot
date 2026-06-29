import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os
import time
import io

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def embed_with_retry(chunks, embeddings, max_retries=5):
    """
    Embeds chunks in small batches with automatic retry + backoff
    so a brief quota hit doesn't crash the app.
    """
    BATCH_SIZE = 20  # embed 20 chunks at a time to stay under rate limits
    all_texts = chunks
    vectors = []

    for i in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[i: i + BATCH_SIZE]
        for attempt in range(max_retries):
            try:
                batch_vectors = embeddings.embed_documents(batch)
                vectors.extend(batch_vectors)
                # Small pause between batches to avoid bursting quota
                if i + BATCH_SIZE < len(all_texts):
                    time.sleep(1)
                break  # success — move to next batch
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s ...
                    st.warning(
                        f"⏳ Rate limit hit. Waiting {wait}s before retrying "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                else:
                    raise  # non-quota error — raise immediately
        else:
            raise RuntimeError(
                "Max retries exceeded. Please wait a minute and re-upload your PDF."
            )
    return vectors


@st.cache_resource(show_spinner="Reading & embedding your PDF... (only done once per file)")
def build_vector_store(file_id: str, file_bytes: bytes):
    """
    Builds the FAISS vector store from the uploaded PDF.
    Cached by (file_id) so embeddings are computed ONCE per unique PDF,
    never re-run on subsequent Streamlit reruns.
    """
    my_pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()

    # Larger chunks = far fewer API calls (stays well within free-tier limits)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150, length_function=len
    )
    chunks = splitter.split_text(text)
    st.info(f"📄 PDF split into {len(chunks)} chunks. Embedding now...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Get vectors with retry/backoff
    vectors = embed_with_retry(chunks, embeddings)

    # Build FAISS index from pre-computed vectors
    vector_store = FAISS.from_embeddings(
        list(zip(chunks, vectors)), embeddings
    )
    return vector_store


st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

if file is not None:
    file_bytes = file.read()
    file_id = f"{file.name}_{len(file_bytes)}"

    # Only embeds once per unique PDF — all reruns use the cache
    vector_store = build_vector_store(file_id, file_bytes)

    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query, k=6)

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        customized_prompt = ChatPromptTemplate.from_template(
            """You are Spandana's personal AI tutor — knowledgeable, clear, and helpful.

            You have access to notes from Spandana's PDF (provided below as context).
            Your job is to answer Spandana's question in the best possible way by:
            1. Using the PDF notes as your PRIMARY source of information.
            2. Supplementing with your own general knowledge to make the answer more complete, 
               clear, and useful — just like how Google Gemini or ChatGPT would answer.
            3. Structuring your response well: use headings, bullet points, numbered lists, 
               bold text, and examples wherever they improve clarity.
            4. If the PDF has relevant content, reference it. If it doesn't cover the topic 
               at all, answer from your general knowledge but mention the PDF didn't cover it.
            5. Only say "I don't know Spandana" if you genuinely have no knowledge about 
               the topic from either the PDF or your training.

            PDF Context:
            {context}

            Question: {input}

            Answer in a well-structured, easy-to-understand format:
            """
        )

        def format_docs(docs):
            return "\n\n".join(
                doc.page_content if isinstance(doc, Document) else str(doc)
                for doc in docs
            )

        chain = (
            {
                "context": lambda _: format_docs(matching_chunks),
                "input": RunnablePassthrough()
            }
            | customized_prompt
            | llm
            | StrOutputParser()
        )

        output = chain.invoke(user_query)
        st.write(output)