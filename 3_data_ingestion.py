# pip install pandas numpy faiss-cpu openai pyarrow
import os
import json
import math
import uuid
import re
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-small"  # 1536-dim, inexpensive
BATCH = 256
OUT_DIR = "rag_store"
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Helpers
# ----------------------------
def _to_list(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]

def build_doc_text(row: pd.Series) -> str:
    parts = [
        f"Title: {row.get('title','')}",
        f"Company: {row.get('company','')}",
        f"Primary category: {row.get('primary_category','')}",
        f"Secondary: {', '.join(row.get('secondary_categories', []) or [])}",
        f"Tags: {', '.join(_to_list(row.get('application_tags','')) + _to_list(row.get('tools_tags','')) + _to_list(row.get('techniques_tags','')) + _to_list(row.get('extra_tags','')))}",
        # f"Short: {row.get('short_summary','')}",
        f"Full: {row.get('full_summary','')}",
        f"URL: {row.get('source_url','')}",
    ]
    return "\n".join([p for p in parts if p and p != "URL: "])

def extract_scores(row: pd.Series) -> dict:
    scores = {}
    for c in row.index:
        if c.startswith("score__"):
            scores[c.replace("score__", "")] = row[c]
        # elif c.startswith("scores."):
        #     scores[c.replace("scores.", "")] = row[c]
    # drop NaNs
    return {k: int(v) if pd.notna(v) else None for k, v in scores.items() if pd.notna(v)}

def build_tags(row: pd.Series, scores: dict) -> list:
    tags = set()
    # high-level tags
    tags.add(f"category:{row.get('primary_category','')}".strip(":"))
    for s in (row.get("secondary_categories") or []):
        tags.add(f"category_secondary:{s}")
    # from tag columns
    for col in ("application_tags", "tools_tags", "techniques_tags", "extra_tags"):
        for t in _to_list(row.get(col)):
            tags.add(t)
            tags.add(f"{col}:{t}")
    # scores as tags
    for k, v in scores.items():
        tags.add(f"score.{k}:{v}")
        # optional “buckets” e.g. >=4
        if isinstance(v, (int, float)):
            if v >= 4: tags.add(f"score.{k}:high")
            if v == 5: tags.add(f"score.{k}:top")
    # newsletter_score
    ns = row.get("newsletter_score")
    if pd.notna(ns):
        ns = int(ns)
        tags.add(f"newsletter_score:{ns}")
        if ns >= 4: tags.add("newsletter_score:high")
    # year_week
    yw = row.get("year_week")
    if pd.notna(yw):
        tags.add(f"year_week:{yw}")
    return sorted(tags)

def parse_datetime(s) -> tuple[str, int]:
    # ISO date string + epoch seconds for range filters
    if pd.isna(s) or not str(s).strip():
        return "", 0
    dt_obj = pd.to_datetime(s, utc=True)
    return dt_obj.isoformat(), int(dt_obj.timestamp())

def embed_texts(texts: list[str], model=EMBED_MODEL, batch=BATCH) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend([e.embedding for e in resp.data])
    arr = np.array(vecs, dtype=np.float32)
    # cosine: normalize to unit length and use IndexFlatIP
    faiss.normalize_L2(arr)
    return arr

# ----------------------------
# Ingest: DataFrame -> FAISS + metadata
# ----------------------------
def ingest_dataframe(df: pd.DataFrame, text_col="__doc_text__"):
    work = df.copy()

    # normalize list column
    # if "secondary_categories" in work.columns and work["secondary_categories"].dtype == object:
    #     work["secondary_categories"] = work["secondary_categories"].apply(
    #         lambda v: v if isinstance(v, list)
    #         else (json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else [])
    #     )

    work[text_col] = work.apply(build_doc_text, axis=1)
    work["tags"] = work.apply(lambda r: build_tags(r, extract_scores(r)), axis=1)
    work["created_at_iso"], work["created_at_epoch"] = zip(*work["created_at"].apply(parse_datetime))
    work["doc_id"] = work.apply(lambda r: f"{r.get('_row_index')}-{uuid.uuid4().hex[:8]}", axis=1)

    # embed
    vectors = embed_texts(work[text_col].tolist())

    # faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    id_index = faiss.IndexIDMap2(index)
    ids = np.arange(len(work), dtype=np.int64)
    id_index.add_with_ids(vectors, ids)

    # persist faiss + vectors
    faiss.write_index(id_index, f"{OUT_DIR}/faiss.index")
    np.save(f"{OUT_DIR}/embeddings.npy", vectors)

    # metadata WITHOUT newsletter_score or any scores.* columns
    meta_cols = [
        "doc_id", "title", "company", "source_url",
        "primary_category",# "secondary_categories", , "_row_index"
        "tags", "created_at_iso", "created_at_epoch", #"year_week"
    ]

    # proactively drop unwanted columns before save (safe if missing)
    drop_cols = ["newsletter_score"] + [c for c in work.columns if c.startswith("scores.") or c.startswith("score__")]
    work = work.drop(columns=drop_cols, errors="ignore")

    work[meta_cols].to_parquet(f"{OUT_DIR}/metadata.parquet", index=False)
    print(f"Ingested {len(work)} docs into ./{OUT_DIR}")
# ----------------------------
# Filtered search (date + score tags)
# ----------------------------
def load_store():
    index = faiss.read_index(f"{OUT_DIR}/faiss.index")
    vecs = np.load(f"{OUT_DIR}/embeddings.npy")
    meta = pd.read_parquet(f"{OUT_DIR}/metadata.parquet")
    return index, vecs, meta

def _candidate_mask(meta: pd.DataFrame,
                    date_from: str|None,
                    date_to: str|None,
                    min_scores: dict|None,
                    include_tags: list[str]|None,
                    include_categories: list[str]|None):
    mask = pd.Series(True, index=meta.index)

    # date range
    if date_from:
        ts = int(pd.to_datetime(date_from, utc=True).timestamp())
        mask &= meta["created_at_epoch"] >= ts
    if date_to:
        ts = int(pd.to_datetime(date_to, utc=True).timestamp())
        mask &= meta["created_at_epoch"] <= ts

    # tag inclusion
    if include_tags:
        want = set(include_tags)
        mask &= meta["tags"].apply(lambda ts: bool(set(ts) & want))

    # category filter
    if include_categories:
        mask &= meta["primary_category"].isin(set(include_categories))

    # min_scores via tags (no score columns stored)
    if min_scores:
        def meets_threshold(tags: list[str]) -> bool:
            # Parse tags like "score.cool_use_cases:5"
            # Accept numeric >= threshold or bucket tags high/top when threshold >=4.
            buckets = set(t for t in tags if ":high" in t or ":top" in t)
            num_map = {}
            for t in tags:
                m = re.match(r"^score\.([a-zA-Z0-9_]+):(\d+)$", t)
                if m:
                    num_map.setdefault(m.group(1), set()).add(int(m.group(2)))

            for key, thr in min_scores.items():
                thr = int(thr)
                nums = num_map.get(key, set())
                if any(n >= thr for n in nums):
                    continue
                # bucket fallbacks for thresholds >=4
                if thr >= 4 and (f"score.{key}:high" in tags or f"score.{key}:top" in tags):
                    continue
                return False
            return True

        mask &= meta["tags"].apply(meets_threshold)

    return mask

def search(query: str, k: int = 10,
           date_from: str|None = None,
           date_to: str|None = None,
           min_scores: dict|None = None,
           include_tags: list[str]|None = None,
           include_categories: list[str]|None = None):
    index, vecs, meta = load_store()
    # embed query
    qv = embed_texts([query])
    # candidate filter
    mask = _candidate_mask(meta, date_from, date_to, min_scores, include_tags, include_categories)
    cand = meta[mask].reset_index(drop=True)

    if len(cand) == 0:
        return []

    # build a tiny sub-index over candidates to enforce the filter strictly
    cand_vecs = vecs[cand.index.values]
    sub = faiss.IndexFlatIP(cand_vecs.shape[1])
    sub.add(cand_vecs)

    # search
    D, I = sub.search(qv, min(k, len(cand)))
    I = I[0]; D = D[0]

    out = []
    for rank, (i_sub, score) in enumerate(zip(I, D), 1):
        row = cand.iloc[int(i_sub)]
        out.append({
            "rank": rank,
            # "score": float(score),  # cosine similarity
            "doc_id": row["doc_id"],
            "title": row["title"],
            "company": row["company"],
            "url": row["source_url"],
            "created_at": row["created_at_iso"],
            "primary_category": row["primary_category"],
            "tags": row["tags"],
            # "scores": row["scores_map"],
        })
    return out

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    df = pd.read_csv("news_taxonomy_labeled.csv")
    ingest_dataframe(df)

    # Query examples
    # results = search(
    #     query="java resource leak fixes with LLM and AST",
    #     k=5,
    #     date_from="2025-05-01",
    #     min_scores={"cool_use_cases": 4},
    #     include_categories=["cool_use_cases"],
    #     include_tags=["java", "score.cool_use_cases:top"]
    # )
    # for r in results:
    #     print(r)