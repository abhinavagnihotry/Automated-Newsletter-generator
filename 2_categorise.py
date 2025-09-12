#!/usr/bin/env python3

# filepath: /Users/abhinavagnihotry/Code/odido/odido_assignment_abhinav/2_categorise.py

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Config
# ---------------------------
MODEL = "gpt-4o-mini"  # good price/quality for routing
BATCH_WINDOW = "24h"   # required by Batch API
INPUT_CSV = Path("filtered_llmops_database.csv")
INPUT_JSONL = Path("news_taxonomy_requests.jsonl")
OUTPUT_JSONL = Path("news_taxonomy_results.jsonl")
OUTPUT_CSV = Path("news_taxonomy_labeled.csv")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")

client = OpenAI()

# ---------------------------
# Taxonomy (your catalog)
# ---------------------------
CATALOG: Dict[str, Any] = {
    "categories": [
        {
            "key": "research_highlights",
            "name": "Research Highlights",
            "definition": "New papers, preprints, benchmarks, datasets, SOTA claims, academic results.",
            "example_signals": [
                "arXiv",
                "preprint",
                "benchmark",
                "dataset",
                "NeurIPS",
                "ICLR",
                "Nature",
                "SOTA",
            ],
        },
        {
            "key": "industry_news",
            "name": "Industry News",
            "definition": "Funding, acquisitions, partnerships, regs/policy, leadership changes, earnings, market moves.",
            "example_signals": [
                "raises",
                "acquired",
                "partnership",
                "announces partnership",
                "regulation",
                "EU AI Act",
                "SEC filing",
            ],
        },
        {
            "key": "cool_use_cases",
            "name": "Cool Use Cases",
            "definition": "Real deployments, case studies, measurable impact, pilots, customer rollouts.",
            "example_signals": [
                "in production",
                "case study",
                "pilot",
                "rollout",
                "customers",
                "ROI",
                "impact",
            ],
        },
        {
            "key": "engineering_deep_dives",
            "name": "Engineering Deep Dives",
            "definition": "Concrete system/infra write-ups on scaling, latency, reliability, inference, distillation, orchestration.",
            "example_signals": [
                "scaling",
                "latency",
                "throughput",
                "GPU",
                "distillation",
                "load balancing",
                "traffic management",
                "observability",
            ],
        },
        {
            "key": "product_launches_tools",
            "name": "Product Launches & Tools",
            "definition": "New products, features, SDKs, APIs, GA/preview, open-source releases.",
            "example_signals": [
                "launch",
                "introducing",
                "announces",
                "SDK",
                "API",
                "general availability",
                "open source",
                "release notes",
            ],
        },
    ]
}

# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PROMPT = (
    "You are an editor routing tech news into newsletter sections. "
    "Be decisive, concise, and consistent with the provided category definitions. "
    "When unsure, still score all categories in [1, 5]. Favor the most specific category. "
    "Return STRICT JSON that matches the JSON schema and nothing else."
)


def build_user_prompt(article: Dict[str, Any]) -> str:
    """
    article = {
      content, tags, url, company, created_at
    }
    """
    schema = (
        "{\n"
        '  "primary_category": "string",\n'
        '  "secondary_categories": ["string", ...],\n'
        '  "scores": {"category_key": <score>, "...": <score>},\n'
        '  "newsletter_score": <score>\n'
        "}"
    )
    return (
        "You will receive:\n"
        "1) Article fields (content, tags, url, company, created_at).\n"
        "2) Category catalog (key, definition, example_signals).\n\n"
        "Tasks:\n"
        "A) Score every category from 0 to 5 based on evidence in the article.\n"
        "B) Pick the best category. Provide up to 2 secondaries if close.\n"
        "JSON schema (return exactly this):\n"
        + schema
        + "\n\n"
        "Article:\n"
        + json.dumps(article, ensure_ascii=False)
        + "\n\n"
        "Categories:\n"
        + json.dumps(CATALOG, ensure_ascii=False)
    )


# ---------------------------
# Helpers
# ---------------------------

def _split_tags(s: Any) -> List[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]


def build_tasks_from_df(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    df = df.reset_index(drop=False).rename(columns={"index": "_row_index"})
    tasks: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        article = {
            "content": row.get("full_summary"),
            "tags": list(
                dict.fromkeys(  # de-duplicate while preserving order
                    _split_tags(row.get("application_tags", ""))
                    + _split_tags(row.get("tools_tags", ""))
                    + _split_tags(row.get("techniques_tags", ""))
                    + _split_tags(row.get("extra_tags", ""))
                )
            ),
            "url": row.get("source_url"),
            "company": row.get("company"),
            "created_at": row.get("created_at"),
        }

        custom_id = f"row-{row['_row_index']}"
        user_prompt = build_user_prompt(article)

        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},  # JSON Mode
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        }
        tasks.append(task)

    return tasks, df


def write_jsonl(objects: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def submit_batch(input_jsonl: Path) -> Any:
    batch_file = client.files.create(file=input_jsonl.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_WINDOW,
    )
    logging.info("Submitted batch id=%s status=%s", batch.id, batch.status)
    return batch


def wait_for_batch(batch_id: str, poll_seconds: int = 15):
    while True:
        b = client.batches.retrieve(batch_id)
        logging.info("status=%s", b.status)
        if b.status in ("completed", "failed", "expired", "cancelling", "cancelled"):
            return b
        time.sleep(poll_seconds)


def download_results(batch: Any, output_path: Path) -> None:
    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")
    result_bytes = client.files.content(batch.output_file_id).content
    output_path.write_bytes(result_bytes)
    logging.info("Saved results to %s", output_path)


# ---------------------------
# Simple parser: read JSONL -> normalize -> merge
# ---------------------------

def parse_batch_results(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_json(path, lines=True)

    def ok(resp: Dict[str, Any]) -> bool:
        try:
            return resp.get("status_code", 0) == 200
        except Exception:
            return False

    ok_rows = raw[raw["response"].apply(ok)].copy()

    def to_parsed(resp: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = resp["body"]["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            return {"_error": f"parse_error: {e}"}

    ok_rows["parsed"] = ok_rows["response"].apply(to_parsed)
    ok_rows["row_index"] = (
        ok_rows["custom_id"].str.extract(r"row-(\d+)", expand=False).astype(int)
    )

    parsed_flat = pd.json_normalize(ok_rows["parsed"])
    scores_flat = (
        pd.json_normalize(ok_rows["parsed"].apply(lambda d: d.get("scores", {}))).add_prefix("score__")
    )

    out = (
        pd.concat([ok_rows[["row_index"]], parsed_flat, scores_flat], axis=1)
        .rename(
            columns={
                "primary_category": "primary_category",
                "secondary_categories": "secondary_categories",
                "newsletter_score": "newsletter_score",
                "scores": "scores_raw",
            }
        )
    )

    err_mask = ~raw["response"].apply(ok) | raw.get("error", pd.Series([None] * len(raw))).notna()
    errors = raw[err_mask][["custom_id", "response"]].copy()
    if "error" in raw.columns:
        errors["error"] = raw["error"]

    return out, errors


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    tasks, df_with_idx = build_tasks_from_df(df)
    write_jsonl(tasks, INPUT_JSONL)
    logging.info("Wrote %d tasks to %s", len(tasks), INPUT_JSONL)

    batch = submit_batch(INPUT_JSONL)
    batch = wait_for_batch(batch.id)

    download_results(batch, OUTPUT_JSONL)

    parsed_df, errors_df = parse_batch_results(OUTPUT_JSONL)

    df_merged = df_with_idx.merge(
        parsed_df, left_on="_row_index", right_on="row_index", how="left"
    ).drop(columns=["row_index"])  # keep original order/index

    df_merged.drop(columns=["_row_index"]).to_csv(OUTPUT_CSV, index=False)
    logging.info("Saved labeled CSV to %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()