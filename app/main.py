"""FastAPI application for the SHL Assessment Recommender.

Endpoints:
    GET  /health  — Readiness check
    POST /chat    — Stateless conversational agent
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import Agent
from app.catalog import load_catalog
from app.config import MAX_TURNS
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import RetrievalEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the application
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
)

# CORS — allow all origins for evaluation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state — initialized on startup
agent: Agent | None = None


@app.on_event("startup")
async def startup() -> None:
    """Load catalog and initialize the agent on startup."""
    global agent
    logger.info("Starting SHL Assessment Recommender...")

    start = time.time()

    # Load catalog
    catalog = load_catalog()
    logger.info("Catalog loaded: %d assessments", len(catalog))

    # Build retrieval engine
    retrieval_engine = RetrievalEngine(catalog)
    logger.info("Retrieval engine ready")

    # Initialize agent
    agent = Agent(retrieval_engine)
    logger.info("Agent initialized in %.1fs", time.time() - start)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint. Returns 200 with status ok when ready."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a conversation and return the next agent response.

    The API is stateless — every call carries the full conversation history.
    No per-conversation state is stored on the server.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialized")

    # Validate input
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    # Check turn limit
    if len(request.messages) > MAX_TURNS:
        logger.warning(
            "Conversation exceeds turn limit: %d > %d",
            len(request.messages),
            MAX_TURNS,
        )
        # Still process but note the limit
        pass

    # Validate message structure
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {msg.role}. Must be 'user' or 'assistant'.",
            )

    start = time.time()
    response = await agent.process(request.messages)
    elapsed = time.time() - start

    logger.info(
        "Chat response in %.1fs — recs: %d, end: %s",
        elapsed,
        len(response.recommendations),
        response.end_of_conversation,
    )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
