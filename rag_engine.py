"""
rag_engine.py
────────────────────────────────────────────────────────────────────────────
Real Retrieval-Augmented Generation system.
• Vector store : ChromaDB (persistent, local)
• Embeddings   : sentence-transformers all-MiniLM-L6-v2  (~80 MB, GPU/CPU)
• Knowledge    : Dynamically loads and chunks Markdown (.md) documents.
• Returns      : top-k chunks + source citations with confidence scores

FIX: SIMILARITY_THRESH corrected from 0.80 → 0.40.
     ChromaDB returns *cosine distance* in [0, 2] (lower = more similar).
     The old threshold of 0.80 was far too permissive — it passed through
     chunks with only ~60% similarity.  0.40 corresponds to a cosine
     similarity of ≥ 0.80, which is a meaningful relevance gate.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from nutrition_knowledge import NUTRITION_FILES

# ── Configuration ──────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "nutrition_knowledge"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
TOP_K_DEFAULT      = 5

# FIX: ChromaDB cosine *distance* is in [0, 2]; lower = more similar.
# 0.40 distance ≈ 0.80 cosine similarity — a meaningful relevance gate.
# The old value of 0.80 passed chunks with only ~60% similarity (too noisy).
SIMILARITY_THRESH  = 0.40

# Chunking configuration
CHUNK_SIZE_WORDS   = 250    # Approx words per chunk (suits MiniLM context window)
CHUNK_OVERLAP      = 50     # Overlap to preserve context between chunks


def _doc_id(doc_text: str) -> str:
    """Stable deterministic ID for a document chunk."""
    return hashlib.md5(doc_text.encode("utf-8")).hexdigest()[:12]


# ── Markdown Processing Helpers ────────────────────────────────────────────

def _extract_text_from_markdown(md_path: str) -> str:
    """Reads raw text from a given Markdown file."""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Error reading Markdown file {md_path}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping word chunks to fit the embedding model."""
    words  = text.split()
    chunks = []

    if not words:
        return chunks

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


# ── Singleton helpers ──────────────────────────────────────────────────────
_collection: Optional[chromadb.Collection] = None
_embed_model: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating + populating it if needed."""
    global _collection
    if _collection is not None:
        return _collection
    
    # # Note: No embedding function passed to ChromaDB; because by default it uses the same 
    # # sentence-transformers model for embedding as we do in _get_embed_model().  
    # Use sentence-transformers embedding function
    # ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    #     model_name=EMBEDDING_MODEL
    # )

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    _populate_collection(_collection)
    return _collection


def _populate_collection(col: chromadb.Collection) -> None:
    """Read Markdown files, chunk them, and upsert into the vector store."""
    existing_ids = set(col.get()["ids"])

    ids, docs, metas = [], [], []

    print("Checking knowledge base for Markdown file updates...")

    for file_info in NUTRITION_FILES:
        file_path   = Path(file_info["directory"])
        source_name = file_info["source"]

        if not file_path.exists():
            print(f"⚠️ Warning: Could not find Markdown file at {file_path}")
            continue

        full_text = _extract_text_from_markdown(str(file_path))
        chunks    = _chunk_text(full_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            doc_id = f"{_doc_id(chunk_text)}_chunk_{i}"

            if doc_id in existing_ids:
                continue

            ids.append(doc_id)
            docs.append(chunk_text)
            metas.append(
                {
                    "source":   source_name,
                    "category": "markdown_document",
                    "doc_id":   doc_id,
                }
            )

    if ids:
        print(f"Adding {len(ids)} new chunks to the database...")
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            col.upsert(
                ids=ids[i:i + batch_size],
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
            )
        print("Knowledge base updated successfully.")
    else:
        print("Knowledge base is already up to date.")


# ── Public API ─────────────────────────────────────────────────────────────

def add_custom_document(content: str, source: str, category: str = "custom") -> str:
    """
    Add a user-supplied document (e.g. pasted text) to the knowledge base.
    Returns the generated doc ID.
    """
    col    = _get_collection()
    doc_id = "custom_" + hashlib.md5(content.encode()).hexdigest()[:10]
    col.upsert(
        ids=[doc_id],
        documents=[content],
        metadatas=[{"source": source, "category": category, "doc_id": doc_id}],
    )
    return doc_id


def retrieve(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Filters by SIMILARITY_THRESH (cosine distance ≤ 0.40).
    """
    col     = _get_collection()
    n_query = min(top_k, col.count())
    if n_query == 0:
        return []

    results = col.query(
        query_texts=[query],
        n_results=n_query,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks

    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        # cosine distance in [0, 2]; score ≈ 1 − dist/2 for normalised vectors
        score = max(0.0, 1.0 - dist / 2.0)
        chunks.append(
            {
                "content":  doc,
                "source":   meta.get("source",   "Unknown"),
                "category": meta.get("category", "general"),
                "distance": round(dist, 4),
                "score":    round(score, 4),
                "citation": f"[{i + 1}]",
            }
        )

    # FIX: filter by corrected threshold (distance ≤ 0.40 → similarity ≥ 0.80)
    chunks = [c for c in chunks if c["distance"] <= SIMILARITY_THRESH]
    return chunks


def retrieve_as_string(query: str, top_k: int = TOP_K_DEFAULT) -> tuple[str, list[dict]]:
    """
    Convenience wrapper: returns (formatted_context_string, raw_chunks).
    The context string is ready to inject into a prompt.
    """
    chunks = retrieve(query, top_k)
    if not chunks:
        return (
            "No highly relevant documents found in the knowledge base for this query. "
            "Apply general evidence-based nutritional reasoning.",
            [],
        )

    lines = []
    for c in chunks:
        lines.append(
            f"{c['citation']} [{c['source']}]\n"
            f"{c['content']}\n"
        )

    context_str = "\n---\n".join(lines)
    return context_str, chunks


def get_kb_stats() -> dict:
    """Return knowledge-base statistics for display in the UI."""
    try:
        col = _get_collection()
        return {
            "total_documents":  col.count(),
            "collection_name":  COLLECTION_NAME,
            "embedding_model": (
                "all-MiniLM-L6-v2 (ONNX)" 
                if hasattr(col, "_embedding_function") and col._embedding_function.name() == "default" 
                else col._embedding_function.name() if hasattr(col, "_embedding_function") else "unknown"
                ),#EMBEDDING_MODEL,
            "persist_dir":      CHROMA_PERSIST_DIR,
            "similarity_thresh": SIMILARITY_THRESH,
        }
    except Exception as e:
        return {"error": str(e)}


def init_rag() -> dict:
    """
    Explicitly initialise the RAG engine (call at app startup).
    Returns stats dict.
    """
    try:
        _get_collection()
        return get_kb_stats()
    except Exception as e:
        return {"error": str(e)}
