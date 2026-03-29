# AI-Assisted Data Insight Generator (GenAI Project)

Data analysts spend significant time manually exploring raw datasets, inspecting schema quality, and translating technical findings into business-friendly narratives.

This project addresses that gap by building an AI-Assisted Data Insight Generator that integrates Large Language Models with Retrieval-Augmented Generation (RAG) to automatically produce:
- dataset summaries,
- feature engineering suggestions,
- business and analytical insights.

The system accepts dataset metadata and retrieval context, then generates grounded, readable insights for analysts, product managers, and business stakeholders.

## Overview
The solution combines a robust preprocessing workflow with a reusable RAG engine.

At a high level:
1. Raw data is cleaned, standardized, enriched, and transformed.
2. Profiling metadata and retrieval chunks are generated.
3. A RAG pipeline retrieves the most relevant chunks for each user question.
4. An LLM produces concise, business-friendly outputs constrained by retrieved evidence.

The current implementation supports:
- Olist-focused preprocessing and metadata generation.
- A working RAG engine for curated chunk files.
- A dynamic dataset engine that profiles any supported dataset and builds an in-memory retrieval store per session.

## Quick Start

**1. Clone & setup environment**
```bash
git clone <your-repo-url>
cd geak-minds-project
python -m venv .venv_win
.venv_win\Scripts\activate
pip install -r requirements.txt
```

**2. Configure API keys**
Create/update `.env` file:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
FINNHUB_API_KEY=your_finnhub_key
```
Get free API keys:
- [Groq API](https://console.groq.com/keys)
- [Google Gemini API](https://aistudio.google.com/app/apikey)

**3. Run the app (fastest)**
```bash
.venv_win\Scripts\streamlit run app.py
```
Then upload a dataset and ask questions. ✨

**4. Or explore sample data**
```bash
# Quick Titanic analysis
.venv_win\Scripts\python.exe -c "from src.dynamic_dataset_engine import analyse_dataset; analyse_dataset('sample_data/titanic.csv',[{'query':'Give me a complete overview of this dataset','output_type':'summary'}])"
```

For full setup details, see [Setup](#setup) section.

## Project Structure

```
Geak-Minds-Capstone-project/
│
├── app.py                          # Streamlit web UI (main entry point)
├── README.md                       # Full documentation
├── requirements.txt                # Python dependencies
├── .env                            
├── .gitignore                       
│
├── notebooks/
│   └── 01_olist_preprocessing.ipynb   # Data cleaning & feature engineering
│
├── 🔧 src/
│   ├── rag_core.py                    # RAG prompt building & LLM calls
│   ├── rag_engine.py                  # Olist-specific RAG demo
│   └── dynamic_dataset_engine.py      # Generic dataset profiling (any CSV/Excel)
│
├── archive/
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── product_category_name_translation.csv
│   └── synthetic_dataset.csv          # Sample e-commerce data
│
├── processed/
│   └── rag/                           # Generated RAG artifacts
│       ├── context_chunks.jsonl       # Text chunks for retrieval
│       ├── business_brief.txt         # High-level summary
│       └── *_metadata_profile.json    # Column profiles
│
└── sample_data/
    └── titanic.csv                    # Generic dataset for testing
```

## Proposed Architecture

### 1) Data Layer
- Raw input tables from the Olist dataset.
- Processed analytical dataset and daily aggregated features.
- RAG artifacts: schema profiles, context chunk store, and business brief.

### 2) Processing Layer
- Notebook pipeline for cleaning, feature engineering, encoding, and output generation.
- Dynamic profiling logic for arbitrary datasets.

### 3) Retrieval Layer
- Sentence-transformers embeddings using all-mpnet-base-v2.
- FAISS IndexFlatIP for cosine-similarity style retrieval via normalized vectors.

### 4) Generation Layer
- Prompt construction using retrieved chunks.
- LLM response generation via Groq primary and Gemini fallback.

### 5) Delivery Layer
- CLI execution for reproducible runs.
- Structured console outputs with query, type, sources, and generated response.

## Preprocessing Pipeline Details
Main pipeline: notebooks/01_olist_preprocessing.ipynb

### Step 1: Load and inspect
- Loads all required Olist source files.
- Validates presence and prints shape, dtypes, nulls, duplicates.

### Step 2: Missing value handling
- Drops rows with critical null keys (order status/customer/product IDs).
- Imputes product_category_name with unknown.
- Imputes review_score by category median with global fallback.
- Tracks imputation flags for transparency.

### Step 3: Datetime normalization
- Parses *_date, *_timestamp, *_at columns to UTC.
- Creates order_purchase_date for daily alignment.

### Step 4: Outlier and anomaly handling
- Removes duplicate orders by order_id.
- Caps payment_value and freight_value at p99.
- Removes unrealistic delivery-delay records (less than -30 days).

### Step 5: Geospatial enrichment
- Aggregates geolocation by zip prefix.
- Enriches customer and seller tables with lat/lng/state context.

### Step 6: Unified merge
- Left-joins orders with items, products, payments, reviews, customers, and sellers.
- Preserves order coverage while adding feature richness.

### Step 7: Feature engineering
Engineered features include:
- delivery_delay_days
- order_processing_time
- is_late_delivery
- items_per_order
- payment_installments_ratio
- review_sentiment
- product_volume_cm3

### Step 8: Encoding
- Label encoding for order_status, review_sentiment, and payment_type.
- One-hot encoding for top product categories with other bucket.

### Step 9: Normalization
- Min-max scaling for value/volume metrics.
- Z-score scaling for time-based performance metrics.

### Step 10: Time-series output
- Daily aggregation for total_orders, total_revenue, avg_review_score, late_delivery_rate.
- Continuous UTC daily index for downstream joins.

### Step 11: Persisted outputs
- processed/olist_processed.parquet
- processed/daily_features.csv
- processed/rag/context_chunks.jsonl
- processed/rag/business_brief.txt
- processed/rag/*_metadata_profile.json

## Features Used for Insight Generation
The retrieval and insight generation flow uses multiple feature categories:
- Operational features: delivery_delay_days, order_processing_time, is_late_delivery.
- Revenue and payment features: payment_value_raw, payment_type, payment_installments_ratio.
- Product features: product dimensions, engineered volume, category one-hot signals.
- Customer and seller geo features: customer_lat/lng, seller_lat/lng, state attributes.
- Quality features: null rates, imputation indicators, duplication stats.
- Time-series features: daily order/revenue/review/delivery signals.

## RAG Pipeline

### Core RAG engine
File: src/rag_core.py and src/rag_engine.py

Flow:
1. Load context chunks.
2. Embed chunk text and build FAISS vector index.
3. Retrieve top-k relevant chunks for each user query.
4. Build a structured prompt with strict grounding instructions.
5. Generate response using LLM provider routing.

### Dynamic dataset RAG engine
File: src/dynamic_dataset_engine.py

Flow:
1. Load any supported dataset format (.csv, .xlsx, .parquet, .json).
2. Auto-profile schema, distributions, quality, and domain hints.
3. Generate dataset-level and column-level chunks in memory.
4. Build session vector store (no disk dependency).
5. Answer summary, feature suggestion, and business insight queries.

*(For a detailed visual structure with emojis, see the [Project Structure](#-project-structure) section at the top.)*

## Setup

### 1) Use the correct environment
Use .venv_win only.

In VS Code:
Ctrl+Shift+P -> Python: Select Interpreter -> choose .venv_win\Scripts\python.exe

### 2) Fresh setup
python -m venv .venv_win
.venv_win\Scripts\activate
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

### 3) Configure API keys in .env
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
FINNHUB_API_KEY=your_finnhub_key_here

## How To Run

### A) Run preprocessing
Open notebooks/01_olist_preprocessing.ipynb and run all cells.

### B) Run Olist RAG demo
.venv_win\Scripts\python.exe src/rag_engine.py

### C) Run dynamic engine demo (Olist + Titanic)
.venv_win\Scripts\python.exe src/dynamic_dataset_engine.py

### D) Run only sample Titanic analysis
.venv_win\Scripts\python.exe -c "from src.dynamic_dataset_engine import analyse_dataset; analyse_dataset('sample_data/titanic.csv',[{'query':'Give me a complete overview of this dataset','output_type':'summary'},{'query':'What analytical insights can be drawn from this data?','output_type':'business_insights'}])"

### E) Launch the Streamlit UI
.venv_win\Scripts\streamlit run app.py

The app will open at http://localhost:8501 (or a nearby port if 8501 is busy).

Features:
- Upload any dataset (CSV, XLSX, Parquet, JSON)
-  Choose from preset queries or enter custom questions
- Select output type: summary, feature suggestions, or business insights
- First run loads embeddings (~30 seconds); subsequent queries are instant
- Download results as .txt files
- Professional dark theme with cached session state

## Output Types
Supported generation modes:
- summary
- feature_suggestions
- business_insights

## Error Handling and Reliability Notes
- Unsupported dataset extensions are rejected with clear errors.
- Missing files fail gracefully with explicit messages.
- Large datasets are sampled for profiling efficiency.
- Retrieval uses relevance thresholding to avoid weak context injection.
- Query-level failures are isolated to prevent full run failure.

## Dataset Reference
Brazilian E-Commerce Public Dataset by Olist:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Titanic sample dataset used for generic validation:
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

## Tech Stack
- Data and preprocessing: pandas, numpy, pyarrow, openpyxl
- Embeddings: sentence-transformers (all-mpnet-base-v2)
- Vector store: FAISS (faiss-cpu)
- LLM providers: Groq and Google Gemini
- Web UI: Streamlit (>= 1.35.0)

