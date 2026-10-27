"""
Vulcan RAG pipeline.

The vector store is built ONCE and reused:
  - First call (or warmup()) loads it from disk, or builds and saves it if missing.
  - It is rebuilt automatically when any knowledge file in software/data/ changes.
  - Every later question only embeds the question itself (one small embed call).

Public API (unchanged):
    response_rag(question, user_expression) -> str
    warmup()  # optional: call at startup so the first question isn't slow
"""

import json
import time
import hashlib
import threading
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "software" / "data"

KNOWLEDGE_FILES = [
    DATA_DIR / "introToERC.txt",
    DATA_DIR / "aboutVulcan.txt",
    DATA_DIR / "team_members.txt",
]

CACHE_DIR = DATA_DIR / ".rag_cache"
STORE_PATH = CACHE_DIR / "vectorstore.json"
HASH_PATH = CACHE_DIR / "source_hash.json"

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100   # was 250 (50% overlap) -> fewer, less redundant chunks
TOP_K = 4
MAX_ANSWER_TOKENS = 120   # answers are meant to be 2-3 sentences; stops rambling

PROMPT = PromptTemplate(
    template=(
        "You are Vulcan, a robot built by ERC (Electronics and Robotics Club) "
        "at BITS Pilani Goa.\n"
        "Answer the question using the documents below.\n"
        "The person's current facial expression is: {user_expression}. "
        "Adjust your tone to suit it.\n"
        "Reply in 2-3 short sentences, suitable for speaking aloud.\n"
        "If the documents don't contain the answer, say briefly that you're not sure "
        "rather than making something up.\n\n"
        "Documents:\n{documents}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["question", "documents", "user_expression"],
)

# ---------------------------------------------------------------------
# One-time setup, cached for the life of the process
# ---------------------------------------------------------------------
_lock = threading.Lock()
LAST_TIMING = {}   # seconds spent in each step of the most recent response_rag()
_retriever = None
_chain = None


def _source_hash():
    """Hash of all knowledge files, so edits trigger a rebuild."""
    h = hashlib.sha256()
    for path in KNOWLEDGE_FILES:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(f"{CHUNK_SIZE}-{CHUNK_OVERLAP}-{EMBED_MODEL}".encode())
    return h.hexdigest()


def _load_or_build_store(embed_model):
    current = _source_hash()
    saved = None
    if HASH_PATH.exists():
        try:
            saved = json.loads(HASH_PATH.read_text()).get("hash")
        except Exception:
            saved = None

    if STORE_PATH.exists() and saved == current:
        print("[RAG] Loading saved vector store...")
        return SKLearnVectorStore(
            embedding=embed_model,
            persist_path=str(STORE_PATH),
            serializer="json",
        )

    print("[RAG] Building vector store (one-time, re-done only if data files change)...")
    text = "\n".join(p.read_text(encoding="utf-8") for p in KNOWLEDGE_FILES)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store = SKLearnVectorStore.from_texts(
        texts=chunks,
        embedding=embed_model,
        persist_path=str(STORE_PATH),
        serializer="json",
    )
    store.persist()
    HASH_PATH.write_text(json.dumps({"hash": current}))
    print(f"[RAG] Built and saved {len(chunks)} chunks.")
    return store


def _init():
    global _retriever, _chain
    if _chain is not None:
        return
    with _lock:
        if _chain is not None:
            return
        embed_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        store = _load_or_build_store(embed_model)
        _retriever = store.as_retriever(search_kwargs={"k": TOP_K})

        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
            num_predict=MAX_ANSWER_TOKENS,
            keep_alive="30m",   # keep llama3.2 loaded in memory between questions
        )
        _chain = PROMPT | llm | StrOutputParser()
        print("[RAG] Ready.")


def warmup():
    """Build/load the store and load the LLM into memory before the first question."""
    _init()
    try:
        _chain.invoke({"question": "Hi", "documents": "", "user_expression": "neutral"})
        print("[RAG] LLM warmed up.")
    except Exception as exc:
        print(f"[RAG] Warmup call failed (is Ollama running?): {exc}")


def response_rag(question, user_expression):
    _init()
    t0 = time.perf_counter()
    docs = _retriever.invoke(question)   # embeds the question + vector search
    doc_text = "\n\n".join(d.page_content for d in docs)
    t1 = time.perf_counter()
    answer = _chain.invoke({
        "question": question,
        "documents": doc_text,
        "user_expression": user_expression or "neutral",
    })
    t2 = time.perf_counter()
    LAST_TIMING.clear()
    LAST_TIMING.update(retrieve=t1 - t0, llm=t2 - t1)
    print("[RAG] output:", answer)
    return answer