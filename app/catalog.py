"""Load, process, and index the SHL product catalog."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import KEY_TO_CODE, CATALOG_PATH
from app.models import CatalogItem

logger = logging.getLogger(__name__)


def _compute_test_type(keys: list[str]) -> str:
    """Map catalog key names to abbreviated test type codes."""
    codes = sorted({KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE})
    return ",".join(codes)


def _build_search_text(item: dict) -> str:
    """Build a rich text representation of a catalog item for retrieval.

    Combines name, description, keys, job levels, and languages into a single
    searchable string. This helps both keyword and semantic matching.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
    ]
    # Add duration info if available
    if item.get("duration"):
        parts.append(f"Duration: {item['duration']}")
    # Add adaptive flag
    if item.get("adaptive") == "yes":
        parts.append("adaptive test")
    return " ".join(parts)


def load_catalog(path: Path | None = None) -> list[CatalogItem]:
    """Load and process the SHL product catalog from JSON.

    Args:
        path: Path to the catalog JSON file. Uses default if None.

    Returns:
        List of processed CatalogItem objects.
    """
    catalog_path = path or CATALOG_PATH
    logger.info("Loading catalog from %s", catalog_path)

    with open(catalog_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read(), strict=False)

    items: list[CatalogItem] = []
    for raw in raw_data:
        test_type = _compute_test_type(raw.get("keys", []))
        search_text = _build_search_text(raw)

        item = CatalogItem(
            entity_id=raw.get("entity_id", ""),
            name=raw.get("name", ""),
            link=raw.get("link", ""),
            job_levels=raw.get("job_levels", []),
            languages=raw.get("languages", []),
            duration=raw.get("duration", ""),
            remote=raw.get("remote", "yes"),
            adaptive=raw.get("adaptive", "no"),
            description=raw.get("description", ""),
            keys=raw.get("keys", []),
            test_type=test_type,
            search_text=search_text,
        )
        items.append(item)

    logger.info("Loaded %d catalog items", len(items))
    return items


def format_catalog_item_for_llm(item: CatalogItem) -> str:
    """Format a single catalog item as a compact string for LLM context.

    Returns a structured representation the LLM can reason over.
    """
    langs = ", ".join(item.languages[:3])
    if len(item.languages) > 3:
        langs += f" (+{len(item.languages) - 3} more)"

    parts = [
        f"NAME: {item.name}",
        f"URL: {item.link}",
        f"TEST_TYPE: {item.test_type}",
        f"CATEGORIES: {', '.join(item.keys)}",
        f"JOB_LEVELS: {', '.join(item.job_levels)}",
    ]
    if item.duration:
        parts.append(f"DURATION: {item.duration}")
    if langs:
        parts.append(f"LANGUAGES: {langs}")
    parts.append(f"ADAPTIVE: {item.adaptive}")
    parts.append(f"DESCRIPTION: {item.description}")

    return "\n".join(parts)
