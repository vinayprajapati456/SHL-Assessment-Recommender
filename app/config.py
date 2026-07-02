"""Configuration for the SHL Assessment Recommender."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CATALOG_PATH = DATA_DIR / "shl_product_catalog.json"

# LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Fallback: Google Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# LLM Parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 25  # seconds, leaves buffer for 30s evaluator timeout

# Retrieval Configuration
TOP_K_RETRIEVAL = 30  # candidates to retrieve for LLM
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1

# Conversation limits
MAX_TURNS = 8  # total messages including user + assistant

# Test type abbreviation mapping
KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Simulations": "S",
    "Assessment Exercises": "E",
}

CODE_TO_KEY = {v: k for k, v in KEY_TO_CODE.items()}
