import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import os

import numpy as np


RAG_CHUNKS_PATH = str(
    Path(__file__).parent.parent / "processed" / "rag" / "context_chunks.jsonl"
)
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_PROVIDER = "groq"  # switch to "gemini" as backup
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
TOP_K = 6
MIN_CHUNK_LENGTH = 20


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine similarity on IndexFlatIP."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def load_chunks(jsonl_path: str) -> list[dict]:
    chunks: list[dict] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON on line {line_no}")
                continue

            text = item.get("text")
            if not isinstance(text, str) or len(text.strip()) < MIN_CHUNK_LENGTH:
                continue

            chunks.append(item)

    print(f"Loaded {len(chunks)} chunks from {jsonl_path}")
    return chunks


def build_vector_store(chunks: list[dict]):
    import faiss
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    embeddings = _normalize_rows(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Vector store built: {index.ntotal} vectors, dimension {dim}")
    return index, chunks, embedder


def retrieve(
    query: str,
    index,
    chunks: list[dict],
    embedder,
    top_k: int = TOP_K,
) -> list[dict]:
    query_vec = embedder.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    query_vec = _normalize_rows(query_vec)

    scores, indices = index.search(query_vec, top_k)

    retrieved: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < 0.25:
            continue

        item = dict(chunks[idx])
        item["score"] = float(score)
        retrieved.append(item)

    if retrieved:
        print("Retrieved chunks:")
        for item in retrieved:
            print(
                f"- {item.get('chunk_id', 'unknown')} | "
                f"{item.get('chunk_type', 'unknown')} | "
                f"score={item['score']:.4f}"
            )
    else:
        print("Retrieved chunks: none above relevance threshold")

    return retrieved


def build_prompt(query: str, retrieved_chunks: list[dict], output_type: str) -> str:
    role_block = (
        "[SYSTEM ROLE]\n"
        "You are a senior business analyst assistant. Your job is to generate\n"
        "clear, accurate, business-friendly insights from dataset metadata.\n"
        "You must ONLY use facts from the CONTEXT below. Never invent numbers.\n"
        "Always explain what findings mean for the business in plain English."
    )

    context_lines = ["[CONTEXT]"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        chunk_type = chunk.get("chunk_type", "unknown")
        text = str(chunk.get("text", ""))
        context_lines.append(f"Source {i}: [{chunk_type}] {text}")
    context_block = "\n".join(context_lines)

    task_block = (
        "[TASK]\n"
        "Based on the context above, answer the following:\n"
        f"{query}"
    )

    if output_type == "summary":
        output_format = (
            "[OUTPUT FORMAT]\n"
            "Write a structured dataset summary with these sections:\n"
            "1. What this dataset is (1-2 sentences)\n"
            "2. Scale and coverage (dates, row counts, key entities)\n"
            "3. Data quality notes (nulls, imputed values, outliers handled)\n"
            "4. Key metrics at a glance (revenue, reviews, delivery)\n"
            "5. What analysis this dataset enables"
        )
    elif output_type == "feature_suggestions":
        output_format = (
            "[OUTPUT FORMAT]\n"
            "Suggest exactly 5 new engineered features. For each:\n"
            "Feature name | Logic/formula | Business value\n"
            "Format as a clean table."
        )
    elif output_type == "business_insights":
        output_format = (
            "[OUTPUT FORMAT]\n"
            "Generate 3 business insights. For each insight use this structure:\n"
            "FINDING: <one sentence stat or observation>\n"
            "WHY IT MATTERS: <business impact in plain English>\n"
            "RECOMMENDED ACTION: <one concrete next step>"
        )
    else:
        raise ValueError(
            "output_type must be one of: summary, feature_suggestions, business_insights"
        )

    closing = "Return ONLY the formatted output. No preamble, no apology, no filler."
    return f"{role_block}\n\n{context_block}\n\n{task_block}\n\n{output_format}\n\n{closing}"


def _call_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env")

        genai.configure(api_key=api_key)

        candidate_models = [
            GEMINI_MODEL,
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
        ]

        seen = set()
        for model_name in candidate_models:
            if model_name in seen:
                continue
            seen.add(model_name)

            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.3},
                )
                text = getattr(response, "text", "")
                if text:
                    return text
            except Exception as inner_exc:
                print(f"Gemini model '{model_name}' failed: {inner_exc}")

        raise ValueError("Gemini returned an empty response for all candidate models")
    except Exception as exc:
        print(f"Gemini call failed: {exc}")
        return ""


def _call_groq(prompt: str) -> str:
    try:
        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env")

        client = Groq(api_key=api_key)

        candidate_models = [
            GROQ_MODEL,
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]

        seen = set()
        for model_name in candidate_models:
            if model_name in seen:
                continue
            seen.add(model_name)

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )

                text = ""
                if response.choices and response.choices[0].message:
                    text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception as inner_exc:
                print(f"Groq model '{model_name}' failed: {inner_exc}")

        raise ValueError("Groq returned an empty response for all candidate models")
    except Exception as exc:
        print(f"Groq call failed: {exc}")
        return ""


def call_llm(prompt: str) -> str:
    provider = LLM_PROVIDER.strip().lower()
    if provider not in {"groq", "gemini"}:
        print(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Falling back to groq.")
        provider = "groq"

    if provider == "groq":
        primary = ("groq", _call_groq)
        fallback = ("gemini", _call_gemini)
    else:
        primary = ("gemini", _call_gemini)
        fallback = ("groq", _call_groq)

    primary_name, primary_fn = primary
    fallback_name, fallback_fn = fallback

    result = primary_fn(prompt)
    if result:
        return result

    print(
        f"Warning: primary provider '{primary_name}' failed. "
        f"Retrying with fallback '{fallback_name}'."
    )

    result = fallback_fn(prompt)
    if result:
        return result

    print("LLM call failed on both primary and fallback providers.")
    return ""


def run_insight_pipeline(
    query: str,
    output_type: str,
    jsonl_path: str = RAG_CHUNKS_PATH,
) -> str:
    chunks = load_chunks(jsonl_path)
    if not chunks:
        print("No chunks loaded. Check the JSONL path and content.")
        return "Not enough context found for this query."

    index, chunks, embedder = build_vector_store(chunks)
    retrieved_chunks = retrieve(query, index, chunks, embedder)

    if not retrieved_chunks:
        return "Not enough context found for this query."

    prompt = build_prompt(query, retrieved_chunks, output_type)
    llm_response = call_llm(prompt)

    source_ids = [chunk.get("chunk_id", "unknown") for chunk in retrieved_chunks]

    print("=" * 60)
    print(f"QUERY: {query}")
    print(f"OUTPUT TYPE: {output_type}")
    print(f"SOURCES USED: {', '.join(source_ids)}")
    print("-" * 60)
    print(llm_response)
    print("=" * 60)

    return llm_response