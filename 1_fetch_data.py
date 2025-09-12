"""
Fetch and persist LLMOps news articles from the last week only.

Behavior:
- Pulls the 'train' split from the Hugging Face dataset 'zenml/llmops-database'.
- Keeps only rows from the last 14 rolling days (UTC) based on 'created_at'.
- Drops duplicates by title and source_url, keeping the most recent.
- Filters to specific industries of interest.
- Writes results to 'filtered_llmops_database.csv'.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from datasets import load_dataset


# --- Configuration ---
INDUSTRIES_OF_INTEREST: tuple[str, ...] = (
	"Tech",
	"Telecommunications",
	"Research & Academia",
)
DEFAULT_DAYS: int = 30  # last 30 rolling days
OUTPUT_CSV = "filtered_llmops_database.csv"


# --- Core helpers ---
def load_llmops_dataset() -> pd.DataFrame:
	"""Load the zenml/llmops-database train split into a DataFrame."""
	ds = load_dataset("zenml/llmops-database")
	return ds["train"].to_pandas()


def parse_and_filter_last_n_days(df: pd.DataFrame, *, days: int = DEFAULT_DAYS) -> pd.DataFrame:
	"""Parse created_at to UTC datetime and keep rows within the last N days.

	Rows with invalid or missing created_at are dropped.
	"""
	if "created_at" not in df.columns:
		raise KeyError("Expected column 'created_at' in dataset")

	now_utc = pd.Timestamp.utcnow()
	window_start = now_utc - pd.Timedelta(days=days)

	df = df.copy()
	df["created_at_dt"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
	before_shape = df.shape
	df = df[df["created_at_dt"].notna()]
	df = df[df["created_at_dt"] >= window_start]

	print(
		f"Keeping only articles from the last {days} days: {window_start:%Y-%m-%d %H:%M} to {now_utc:%Y-%m-%d %H:%M} UTC"
	)
	print("Shape after date filter:", df.shape, "(from", before_shape, ")")
	return df


def drop_duplicates_recent(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop duplicates by title and source_url, keeping the most recent row."""
	cols_present = set(df.columns)
	by_title = "title" in cols_present
	by_url = "source_url" in cols_present

	if "created_at_dt" not in cols_present:
		# Fallback sort if created_at_dt missing
		sort_key = "created_at" if "created_at" in cols_present else None
	else:
		sort_key = "created_at_dt"

	res = df.copy()
	if sort_key is not None:
		res = res.sort_values(sort_key)

	before = res.shape
	if by_title:
		res = res.drop_duplicates(subset=["title"], keep="last")
	if by_url:
		res = res.drop_duplicates(subset=["source_url"], keep="last")

	print("Shape after removing duplicates:", res.shape, "(from", before, ")")
	return res


def filter_industries(df: pd.DataFrame, industries: Iterable[str]) -> pd.DataFrame:
	"""Keep only rows whose 'industry' is among the provided industries."""
	if "industry" not in df.columns:
		print("Warning: 'industry' column not found; skipping industry filter.")
		return df

	res = df[df["industry"].isin(list(industries))].copy()
	print("Shape after industry filter:", res.shape)
	return res


def main(days: int = DEFAULT_DAYS) -> None:
	print("Loading dataset 'zenml/llmops-database'…")
	df = load_llmops_dataset()
	print("Initial shape:", df.shape)

	df = parse_and_filter_last_n_days(df, days=days)
	df = drop_duplicates_recent(df)
	df = filter_industries(df, INDUSTRIES_OF_INTEREST)

	df.to_csv(OUTPUT_CSV, index=False)
	print(f"Written {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
