import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

"""RAG engine script entrypoint.

Purpose:
- Provides a single command-line runnable script for the RAG workflow.
- Uses shared logic from rag_core.py (chunk loading, retrieval, prompting, LLM routing).
- Runs demo queries for summary, feature suggestions, and business insights.
"""

from typing import Any

from rag_core import run_insight_pipeline


def main() -> None:
    demos: list[dict[str, Any]] = [
        {
            "query": "Give me a complete overview of this dataset",
            "output_type": "summary",
        },
        {
            "query": (
                "What new features can I engineer from this e-commerce data "
                "to better understand customer behavior and delivery performance?"
            ),
            "output_type": "feature_suggestions",
        },
        {
            "query": (
                "What are the most critical business insights from this dataset "
                "that a product manager should know immediately?"
            ),
            "output_type": "business_insights",
        },
    ]

    for i, item in enumerate(demos, start=1):
        print(f"\n\n### DEMO RUN {i} ###")
        run_insight_pipeline(item["query"], item["output_type"])
        print("\n" + "#" * 60 + "\n")


if __name__ == "__main__":
    main()
