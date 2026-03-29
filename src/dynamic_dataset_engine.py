import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from typing import Any

import numpy as np
import pandas as pd

from rag_core import _normalize_rows, build_prompt, call_llm


EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 6
MIN_CHUNK_LENGTH = 20
MAX_CATEGORIES = 15
SAMPLE_ROWS = 3
MAX_ROWS_PROFILE = 50000


def load_any_dataset(filepath: str) -> pd.DataFrame:
	path = Path(filepath)
	if not path.exists():
		raise FileNotFoundError(f"File not found: {filepath}")

	ext = path.suffix.lower()
	if ext == ".csv":
		df = pd.read_csv(path)
	elif ext == ".xlsx":
		df = pd.read_excel(path)
	elif ext == ".parquet":
		df = pd.read_parquet(path)
	elif ext == ".json":
		df = pd.read_json(path)
	else:
		raise ValueError(
			f"Unsupported file format: {ext}. Supported: .csv, .xlsx, .parquet, .json"
		)

	total_rows = len(df)
	if total_rows > MAX_ROWS_PROFILE:
		print(
			f"Large dataset detected ({total_rows} rows). Profiling on a 50,000 row sample."
		)
		df = df.sample(n=MAX_ROWS_PROFILE, random_state=42)

	print(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
	return df


def _safe_sample_values(series: pd.Series) -> list[Any]:
	non_null = series.dropna()
	if non_null.empty:
		return []
	return non_null.head(SAMPLE_ROWS).tolist()


def _infer_datetime_series(series: pd.Series) -> tuple[bool, pd.Series | None]:
	if pd.api.types.is_datetime64_any_dtype(series):
		parsed = pd.to_datetime(series, errors="coerce")
		return True, parsed

	# Infer datetimes only for text-like columns to avoid misclassifying
	# numeric/boolean fields and to prevent parser errors on non-date values.
	if not (
		pd.api.types.is_object_dtype(series)
		or pd.api.types.is_string_dtype(series)
		or isinstance(series.dtype, pd.CategoricalDtype)
	):
		return False, None

	non_null = series.dropna()
	if non_null.empty:
		return False, None

	sample_size = min(500, len(non_null))
	sample = non_null.sample(n=sample_size, random_state=42)
	parsed_sample = pd.to_datetime(sample, errors="coerce")
	parse_ratio = parsed_sample.notna().mean()
	if parse_ratio >= 0.8:
		parsed_full = pd.to_datetime(series, errors="coerce")
		return True, parsed_full
	return False, None


def auto_profile(df: pd.DataFrame, dataset_name: str) -> dict:
	total_rows = len(df)
	total_cols = len(df.columns)

	columns_profile: list[dict[str, Any]] = []
	n_numeric_cols = 0
	n_categorical_cols = 0
	n_datetime_cols = 0

	for col in df.columns:
		series = df[col]
		null_count = int(series.isna().sum())
		null_rate = float(null_count / total_rows) if total_rows else 0.0
		n_unique = int(series.nunique(dropna=True))
		unique_ratio = (n_unique / total_rows) if total_rows else 0.0

		is_datetime, parsed_datetime = _infer_datetime_series(series)
		is_numeric = (
			pd.api.types.is_numeric_dtype(series)
			and not pd.api.types.is_bool_dtype(series)
			and not is_datetime
		)
		is_categorical = not is_numeric and not is_datetime

		col_profile: dict[str, Any] = {
			"column": col,
			"dtype": "datetime" if is_datetime else str(series.dtype),
			"null_count": null_count,
			"null_rate": null_rate,
			"n_unique": n_unique,
			"sample_values": _safe_sample_values(series),
		}

		if is_numeric:
			non_null = series.dropna()
			if not non_null.empty:
				col_profile["min"] = float(non_null.min())
				col_profile["max"] = float(non_null.max())
				col_profile["mean"] = float(non_null.mean())
				col_profile["median"] = float(non_null.median())
				col_profile["std"] = float(non_null.std(ddof=1)) if len(non_null) > 1 else 0.0
				col_profile["p25"] = float(non_null.quantile(0.25))
				col_profile["p75"] = float(non_null.quantile(0.75))

			n_numeric_cols += 1

		elif is_datetime and parsed_datetime is not None:
			non_null_dt = parsed_datetime.dropna()
			if not non_null_dt.empty:
				dt_min = non_null_dt.min()
				dt_max = non_null_dt.max()
				col_profile["min"] = str(dt_min)
				col_profile["max"] = str(dt_max)
				col_profile["date_range_days"] = int((dt_max - dt_min).days)
			n_datetime_cols += 1

		else:
			non_null = series.dropna()
			if not non_null.empty:
				top_values = non_null.astype(str).value_counts().head(MAX_CATEGORIES)
				col_profile["top_values"] = {k: int(v) for k, v in top_values.items()}
			n_categorical_cols += 1

		is_constant = n_unique == 1
		if is_numeric and "min" in col_profile and "max" in col_profile:
			if col_profile["min"] == col_profile["max"]:
				is_constant = True

		col_profile["is_id_column"] = bool(null_rate == 0 and unique_ratio > 0.95)
		col_profile["is_constant"] = bool(is_constant)

		columns_profile.append(col_profile)

	total_cells = max(total_rows * total_cols, 1)
	total_null_rate = float(df.isna().sum().sum() / total_cells)
	duplicate_row_count = int(df.duplicated().sum()) if total_rows else 0
	duplicate_row_rate = float(duplicate_row_count / total_rows) if total_rows else 0.0
	memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
	n_id_columns = sum(1 for c in columns_profile if c["is_id_column"])
	n_constant_columns = sum(1 for c in columns_profile if c["is_constant"])

	colnames = [str(c).lower() for c in df.columns]
	domain_map = {
		"e-commerce": ["order", "product", "customer", "seller", "payment", "cart"],
		"healthcare": ["patient", "diagnosis", "hospital", "medication", "clinical"],
		"hr": ["salary", "employee", "department", "hire"],
		"financial": ["price", "volume", "open", "close", "ticker"],
		"geospatial": ["lat", "lng", "latitude", "longitude", "city"],
	}

	inferred_domain = "general"
	for domain, keywords in domain_map.items():
		if any(any(keyword in name for keyword in keywords) for name in colnames):
			inferred_domain = domain
			break

	numeric_ratio = (n_numeric_cols / total_cols) if total_cols else 0.0
	if inferred_domain == "general" and n_datetime_cols > 0 and numeric_ratio > 0.70:
		inferred_domain = "time-series"

	summary_stats = {
		"total_null_rate": total_null_rate,
		"n_numeric_cols": n_numeric_cols,
		"n_categorical_cols": n_categorical_cols,
		"n_datetime_cols": n_datetime_cols,
		"n_id_columns": n_id_columns,
		"n_constant_columns": n_constant_columns,
		"memory_usage_mb": memory_usage_mb,
		"duplicate_row_count": duplicate_row_count,
		"duplicate_row_rate": duplicate_row_rate,
	}

	profile = {
		"dataset_name": dataset_name,
		"shape": [int(total_rows), int(total_cols)],
		"columns": columns_profile,
		"summary_stats": summary_stats,
		"inferred_domain": inferred_domain,
	}

	print(f"Inferred domain: {inferred_domain}")
	return profile


def generate_chunks_from_profile(profile: dict) -> list[dict]:
	dataset_name = str(profile["dataset_name"])
	rows, cols = profile["shape"]
	summary = profile["summary_stats"]
	columns = profile["columns"]

	chunks: list[dict[str, str]] = []

	overview_text = (
		f"Dataset '{dataset_name}' appears to be {profile['inferred_domain']}. "
		f"It has {rows:,} rows and {cols:,} columns. "
		f"Columns include {summary['n_numeric_cols']} numeric, "
		f"{summary['n_categorical_cols']} categorical, and {summary['n_datetime_cols']} datetime fields. "
		f"Overall null rate is {summary['total_null_rate']:.2%}. "
		f"Duplicate row rate is {summary['duplicate_row_rate']:.2%}. "
		f"Estimated memory usage is {summary['memory_usage_mb']:.2f} MB. "
		f"Detected {summary['n_id_columns']} ID columns (treat as non-features) and "
		f"{summary['n_constant_columns']} constant columns (likely useless for modeling)."
	)
	chunks.append(
		{
			"chunk_id": f"{dataset_name}_overview_001",
			"source": dataset_name,
			"chunk_type": "dataset_summary",
			"text": overview_text,
		}
	)

	numeric_lines: list[str] = []
	for col in columns:
		if col.get("is_id_column") or col.get("is_constant"):
			continue
		if col.get("null_rate", 0.0) >= 1.0:
			continue
		if "min" in col and "mean" in col and col.get("dtype") != "datetime":
			numeric_lines.append(
				f"{col['column']}: min={col['min']}, max={col.get('max')}, "
				f"mean={col['mean']:.2f}, median={col.get('median', 0.0):.2f}, "
				f"std={col.get('std', 0.0):.2f}, nulls={col['null_rate']:.1%}"
			)
	if numeric_lines:
		chunks.append(
			{
				"chunk_id": f"{dataset_name}_numeric_001",
				"source": dataset_name,
				"chunk_type": "numeric_stats",
				"text": "\n".join(numeric_lines),
			}
		)

	categorical_lines: list[str] = []
	for col in columns:
		if col.get("is_id_column"):
			continue
		if col.get("null_rate", 0.0) >= 1.0:
			continue
		if "top_values" in col and col["top_values"]:
			categorical_lines.append(
				f"{col['column']}: top values = {col['top_values']}, nulls={col['null_rate']:.1%}"
			)
	if categorical_lines:
		chunks.append(
			{
				"chunk_id": f"{dataset_name}_categorical_001",
				"source": dataset_name,
				"chunk_type": "categorical_stats",
				"text": "\n".join(categorical_lines),
			}
		)

	datetime_lines: list[str] = []
	for col in columns:
		if col.get("dtype") != "datetime":
			continue
		if col.get("null_rate", 0.0) >= 1.0:
			continue
		if "min" in col and "max" in col:
			datetime_lines.append(
				f"{col['column']}: {col['min']} to {col['max']}, span = {col.get('date_range_days', 0)} days"
			)
	if datetime_lines:
		chunks.append(
			{
				"chunk_id": f"{dataset_name}_datetime_001",
				"source": dataset_name,
				"chunk_type": "datetime_stats",
				"text": "\n".join(datetime_lines),
			}
		)

	high_null = [c["column"] for c in columns if c.get("null_rate", 0.0) > 0.30]
	near_empty = [c["column"] for c in columns if c.get("null_rate", 0.0) > 0.80]
	constant_cols = [c["column"] for c in columns if c.get("is_constant")]

	total_null_rate = summary["total_null_rate"]
	duplicate_rate = summary["duplicate_row_rate"]
	if total_null_rate < 0.05 and duplicate_rate < 0.01:
		verdict = "good"
	elif total_null_rate < 0.20:
		verdict = "moderate"
	else:
		verdict = "poor"

	quality_text = (
		f"Columns with null_rate > 30% (potentially unreliable): {high_null}. "
		f"Columns with null_rate > 80% (nearly empty — consider dropping): {near_empty}. "
		f"Constant columns (useless for analysis): {constant_cols}. "
		f"Duplicate rows: {summary['duplicate_row_count']} ({duplicate_rate:.2%}). "
		f"Overall data quality verdict: {verdict}."
	)
	chunks.append(
		{
			"chunk_id": f"{dataset_name}_quality_001",
			"source": dataset_name,
			"chunk_type": "data_quality",
			"text": quality_text,
		}
	)

	for col in columns:
		if col.get("is_id_column") or col.get("is_constant"):
			continue
		if col.get("null_rate", 0.0) >= 1.0:
			continue

		clean_name = str(col["column"]).replace(" ", "_")
		base = (
			f"Column '{col['column']}' is {col['dtype']} with {rows:,} rows, "
			f"{col['null_rate']:.1%} nulls, {col['n_unique']:,} unique values. "
		)

		if col.get("dtype") == "datetime" and "min" in col and "max" in col:
			details = (
				f"Range: {col['min']} to {col['max']} "
				f"({col.get('date_range_days', 0)} days). "
			)
		elif "mean" in col:
			details = (
				f"Range: {col.get('min')} to {col.get('max')}, "
				f"mean={col.get('mean', 0.0):.2f}, median={col.get('median', 0.0):.2f}. "
			)
		elif "top_values" in col and col["top_values"]:
			details = f"Top values: {col['top_values']}. "
		else:
			details = ""

		samples = col.get("sample_values", [])
		sample_text = f"Sample values: {', '.join(map(str, samples))}." if samples else ""
		text = f"{base}{details}{sample_text}".strip()

		if len(text) < MIN_CHUNK_LENGTH:
			continue

		chunks.append(
			{
				"chunk_id": f"col_{dataset_name}_{clean_name}",
				"source": dataset_name,
				"chunk_type": "column_detail",
				"text": text,
			}
		)

	print(f"Generated {len(chunks)} chunks for '{dataset_name}'")
	return chunks


def build_session_store(chunks: list[dict], embedder=None) -> tuple:
	if not chunks:
		raise ValueError("No chunks available to build session store.")

	import faiss

	if embedder is None:
		from sentence_transformers import SentenceTransformer
		print("Loading embedding model...")
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

	print(f"Session store ready: {index.ntotal} chunks, dim={dim}")
	return index, chunks, embedder


def session_retrieve(
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

	results: list[dict] = []
	for score, idx in zip(scores[0], indices[0]):
		if idx < 0:
			continue
		if float(score) < 0.25:
			continue
		item = dict(chunks[idx])
		item["score"] = float(score)
		results.append(item)

	if results:
		print("Retrieved chunks:")
		for item in results:
			print(
				f"- {item.get('chunk_id', 'unknown')} | "
				f"{item.get('chunk_type', 'unknown')} | score={item['score']:.4f}"
			)
	else:
		print("Retrieved chunks: none above relevance threshold")

	return results


def analyse_dataset(filepath: str, queries: list[dict]) -> dict[str, str]:
	results: dict[str, str] = {}

	try:
		df = load_any_dataset(filepath)
	except FileNotFoundError as exc:
		print(f"Error: {exc}")
		return results
	except ValueError as exc:
		print(f"Error: {exc}")
		return results
	except Exception as exc:
		print(f"Error while loading dataset: {exc}")
		return results

	dataset_name = Path(filepath).stem

	try:
		profile = auto_profile(df, dataset_name)
		chunks = generate_chunks_from_profile(profile)
		index, session_chunks, embedder = build_session_store(chunks)
	except Exception as exc:
		print(f"Error while profiling or building store: {exc}")
		return results

	for item in queries:
		query_text = str(item.get("query", "")).strip()
		output_type = str(item.get("output_type", "")).strip()

		if output_type not in {"summary", "feature_suggestions", "business_insights"}:
			print(
				f"Error: Invalid output_type '{output_type}'. "
				"Allowed: summary, feature_suggestions, business_insights"
			)
			results[query_text or "<empty_query>"] = "Invalid output_type."
			continue

		if not query_text:
			print("Error: Empty query provided.")
			results["<empty_query>"] = "Query cannot be empty."
			continue

		try:
			retrieved = session_retrieve(query_text, index, session_chunks, embedder)
			if not retrieved:
				results[query_text] = "Not enough context found."
				continue

			prompt = build_prompt(query_text, retrieved, output_type)
			llm_response = call_llm(prompt)
			if not llm_response:
				llm_response = "LLM returned an empty response."

			results[query_text] = llm_response
			sources = ", ".join(chunk["chunk_id"] for chunk in retrieved)

			print("=" * 60)
			print(f"QUERY    : {query_text}")
			print(f"TYPE     : {output_type}")
			print(f"SOURCES  : {sources}")
			print("-" * 60)
			print(llm_response)
			print("=" * 60)

		except Exception as exc:
			print(f"Error while processing query '{query_text}': {exc}")
			results[query_text] = f"Error during analysis: {exc}"

	return results


if __name__ == "__main__":
	Path("sample_data").mkdir(exist_ok=True)

	print("=" * 60)
	print("TEST 1: OLIST DATASET")
	print("=" * 60)
	analyse_dataset(
		filepath="processed/olist_processed.parquet",
		queries=[
			{
				"query": "Give me a complete overview of this dataset",
				"output_type": "summary",
			},
			{
				"query": "What are the most important business insights a product manager should act on immediately?",
				"output_type": "business_insights",
			},
			{
				"query": "What new features should I engineer from this data to improve demand forecasting?",
				"output_type": "feature_suggestions",
			},
		],
	)

	print("=" * 60)
	print("TEST 2: TITANIC DATASET")
	print("=" * 60)
	analyse_dataset(
		filepath="sample_data/titanic.csv",
		queries=[
			{
				"query": "Give me a complete overview of this dataset",
				"output_type": "summary",
			},
			{
				"query": "What analytical insights can be drawn from this data?",
				"output_type": "business_insights",
			},
		],
	)
