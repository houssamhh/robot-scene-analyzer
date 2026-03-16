from __future__ import annotations

from typing import Dict, List, Optional

from demo_data import RAG_DOCS

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


def build_simple_rag_context(question: str, scene_text: str) -> List[Dict[str, str]]:
    """Fallback lexical retrieval so the demo works without external services."""
    query = f"{question} {scene_text}".lower()
    scored = []
    for doc in RAG_DOCS:
        content = doc["content"].lower()
        score = sum(1 for token in query.split() if token in content)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:3]]


def build_langchain_vectorstore() -> Optional[FAISS]:
    if not LANGCHAIN_AVAILABLE:
        return None
    try:
        embeddings = OpenAIEmbeddings()
        texts = [f"{d['title']}\n{d['content']}" for d in RAG_DOCS]
        metadatas = [{"title": d["title"]} for d in RAG_DOCS]
        return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    except Exception:
        return None
