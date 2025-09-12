# pip install pandas numpy faiss-cpu openai pyarrow
import os
import json
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ----------------------------
# Config
# ----------------------------
STORE_DIR = "rag_store"
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Set OPENAI_API_KEY in your environment."
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Category template
# ----------------------------
CATALOG = {
    "categories": [
        {
            "key": "research_highlights",
            "name": "Research Highlights",
            "definition": "New papers, preprints, benchmarks, datasets, SOTA claims, academic results.",
            "example_signals": ["arXiv", "preprint", "benchmark", "dataset", "NeurIPS", "ICLR", "Nature", "SOTA"],
        },
        {
            "key": "industry_news",
            "name": "Industry News",
            "definition": "Funding, acquisitions, partnerships, regs/policy, leadership changes, earnings, market moves.",
            "example_signals": ["raises", "acquired", "partnership", "announces partnership", "regulation", "EU AI Act", "SEC filing"],
        },
        {
            "key": "cool_use_cases",
            "name": "Cool Use Cases",
            "definition": "Real deployments, case studies, measurable impact, pilots, customer rollouts.",
            "example_signals": ["in production", "case study", "pilot", "rollout", "customers", "ROI", "impact"],
        },
        {
            "key": "engineering_deep_dives",
            "name": "Engineering Deep Dives",
            "definition": "Concrete system/infra write-ups on scaling, latency, reliability, inference, distillation, orchestration.",
            "example_signals": ["scaling", "latency", "throughput", "GPU", "distillation", "load balancing", "traffic management", "observability"],
        },
        {
            "key": "product_launches_tools",
            "name": "Product Launches & Tools",
            "definition": "New products, features, SDKs, APIs, GA/preview, open-source releases.",
            "example_signals": ["launch", "introducing", "announces", "SDK", "API", "general availability", "open source", "release notes"],
        },
    ]
}

# ----------------------------
# Store IO
# ----------------------------
def load_store():
    index = faiss.read_index(f"{STORE_DIR}/faiss.index")
    vecs = np.load(f"{STORE_DIR}/embeddings.npy")
    meta = pd.read_parquet(f"{STORE_DIR}/metadata.parquet")
    # safety: ensure list-type column for tags
    if meta["tags"].dtype == object:
        meta["tags"] = meta["tags"].apply(lambda v: v if isinstance(v, list) else [])
    return index, vecs, meta

def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = np.array([e.embedding for e in resp.data], dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr

# ----------------------------
# Filtering helpers (we stored scores as tags)
# ----------------------------
def _ts(x: str) -> int:
    return int(pd.to_datetime(x, utc=True).timestamp())

def _has_min_score(tags: list[str], cat_key: str, thr: int) -> bool:
    # Accept numeric >= thr or buckets for >=4
    if thr >= 4 and (f"score.{cat_key}:high" in tags or f"score.{cat_key}:top" in tags):
        return True
    for n in range(thr, 6):
        if f"score.{cat_key}:{n}" in tags:
            return True
    return False

def candidate_mask(meta: pd.DataFrame,
                   include_category: str | None,
                   date_from: str | None,
                   date_to: str | None,
                #    min_score_for_cat: tuple[str, int] | None,
                #    must_have_tags: list[str] | None
                   ):
    """Create boolean mask for filtering metadata based on criteria."""
    mask = pd.Series(True, index=meta.index)
    
    if include_category:
        mask &= meta["primary_category"].eq(include_category)
    if date_from:
        mask &= meta["created_at_epoch"] >= _ts(date_from)
    if date_to:
        mask &= meta["created_at_epoch"] <= _ts(date_to)
    # if must_have_tags:
    #     need = set(must_have_tags)
    #     mask &= meta["tags"].apply(lambda ts: bool(set(ts) & need))
    # if min_score_for_cat:
    #     key, thr = min_score_for_cat
    #     mask &= meta["tags"].apply(lambda ts: _has_min_score(ts, key, thr))
    return mask

# ----------------------------
# Vector search over filtered candidates
# ----------------------------
def faiss_search(query: str, k: int,
                 include_category: str | None = None,
                 date_from: str | None = None,
                 date_to: str | None = None,
                #  min_score_for_cat: tuple[str, int] | None = None,
                #  must_have_tags: list[str] | None = None
                 ):
    index, vecs, meta = load_store()
    qv = embed_texts([query])

    mask = candidate_mask(meta, include_category, date_from, date_to,
                          #min_score_for_cat, must_have_tags
                          )
    cand = meta[mask].reset_index(drop=True)
    if cand.empty:
        return []

    cand_vecs = vecs[cand.index.values]
    sub = faiss.IndexFlatIP(cand_vecs.shape[1])
    sub.add(cand_vecs)

    D, I = sub.search(qv, min(k, len(cand)))
    I, D = I[0], D[0]

    results = []
    for i_sub, sim in zip(I, D):
        row = cand.iloc[int(i_sub)]
        results.append({
            "sim": float(sim),
            "title": row["title"],
            "url": row["source_url"],
            "company": row.get("company", ""),
            "primary_category": row["primary_category"],
            "tags": row["tags"],
            "created_at": row["created_at_iso"],
            "created_at_epoch": int(row["created_at_epoch"]),
            "doc_id": row["doc_id"],
        })
    return results

# ----------------------------
# Rerank: semantic sim + recency + score tags + category match
# ----------------------------
def rerank(items: list[dict], cat_key: str) -> list[dict]:
    """Rerank items based on semantic similarity, recency, and category scores."""
    if not items:
        return []
    
    # Normalize recency in [0,1]
    epochs = np.array([it["created_at_epoch"] for it in items], dtype=np.float64)
    if epochs.max() == epochs.min():
        rec = np.ones_like(epochs)
    else:
        rec = (epochs - epochs.min()) / (epochs.max() - epochs.min())

    ranked = []
    for it, r in zip(items, rec):
        tags = set(it["tags"])
        score = 0.0
        score += it["sim"] * 100.0          # semantic fit (dominant)
        score += r * 8.0                    # recency boost
        if f"score.{cat_key}:top" in tags:  # strong category signal
            score += 12.0
        elif f"score.{cat_key}:high" in tags:
            score += 6.0
        if it["primary_category"] == cat_key:
            score += 4.0
        # if "newsletter_score:high" in tags:
        #     score += 3.0
        # if "newsletter_score:top" in tags:
        #     score += 6.0
        
        it = dict(it)
        it["rank_score"] = score
        ranked.append(it)

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    return ranked

# ----------------------------
# Build newsletter content (Markdown) with LLM
# ----------------------------
SYS_PROMPT = (
    "You are a sharp newsletter editor. Write concise, scannable copy. "
    "For each section, pick the given articles and summarize in 2–3 tight bullets each. "
    "No hype, no filler. Keep links and companies accurate. Use Markdown."
)

def generate_newsletter_md(sections: dict[str, list[dict]], week_label: str) -> str:
    # Compact context for the model
    ctx = {
        "week": week_label,
        "sections": {
            k: [
                {
                    "title": it["title"],
                    "url": it["url"],
                    "company": it["company"],
                    "created_at": it["created_at"],
                } for it in v
            ] for k, v in sections.items()
        }
    }

    user_prompt = (
        "Template sections are keys; values are lists of selected articles with title, url, company, created_at.\n"
        "Write a Markdown newsletter with H1: 'AI Weekly – {week}', then one H2 per section name.\n"
        "Under each H2, include up to 3 articles. For each article, write 2–3 bullets summarizing what's new and why it matters.\n"
        "No extra preface or outro.\n\n"
        f"SECTIONS JSON:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        "Section names:\n" + ", ".join([c["name"] for c in CATALOG["categories"]])
    )

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return resp.choices[0].message.content.strip()

# ----------------------------
# Orchestrator
# ----------------------------
def build_newsletter(date_from: str, date_to: str | None = None,
                     per_category: int = 3, k_pool: int = 40) -> str:
    """
    Build newsletter for given date range.
    
    Args:
        date_from/date_to: ISO date strings like '2025-05-01'
        per_category: max items to keep per section
        k_pool: how many candidates to pull before rerank
    """
    sections = {}

    for cat in CATALOG["categories"]:
        key = cat["key"]
        name = cat["name"]
        # Simple query: name + signals + definition
        query = f"{name}. Signals: {', '.join(cat['example_signals'])}. {cat['definition']}"
        
        # Pull candidate pool filtered to category and strong category score
        pool = faiss_search(
            query=query,
            k=k_pool,
            include_category=key,
            #date_from=date_from,
            #date_to=date_to,
            #min_score_for_cat=(key, 4),
            #must_have_tags=["newsletter_score:high"],
        )
        
        ranked = rerank(pool, key)
        topn = ranked[:per_category]
        if topn:
            sections[key] = topn

    # Compose week label
    if date_to:
        week_label = f"{date_from} → {date_to}"
    else:
        week_label = f"Week of {date_from}"

    return generate_newsletter_md(sections, week_label)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Last 7 days example (adjust dates to your dataset)
    md = build_newsletter(date_from="2025-05-01", date_to="2025-05-11", per_category=3)
    with open("newsletter.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("Wrote newsletter.md")