"""Conversational agent for SHL Assessment Recommender.

Orchestrates:
1. Query understanding from conversation history
2. Assessment retrieval from the catalog
3. LLM-powered response generation with structured output
4. Response validation and schema compliance
5. Heuristic fallback when no LLM is available
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.catalog import format_catalog_item_for_llm
from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    TOP_K_RETRIEVAL,
)
from app.models import CatalogItem, ChatResponse, Message, Recommendation
from app.prompts import SYSTEM_PROMPT
from app.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

# Off-topic detection patterns
OFF_TOPIC_PATTERNS = [
    r"\b(salary|compensation|pay range|how much to pay)\b",
    r"\b(legal|lawsuit|compliance requirement|regulation)\b",
    r"\b(visa|immigration|work permit)\b",
    r"\b(ignore|forget|disregard).*(instructions|rules|system)\b",
    r"\b(tell me a joke|write a poem|sing a song)\b",
    r"\b(interview questions|how to interview)\b",
]

# Signals that the user wants to end the conversation
CONFIRMATION_PATTERNS = [
    r"\b(perfect|confirmed?|lock(ing)? it in|that'?s? (good|it|what we need|great))\b",
    r"\b(done|thanks|thank you|all set|wrap up)\b",
    r"\b(final list|keep (the|this)|as[- ]is)\b",
]

# Vague query patterns that need clarification
VAGUE_PATTERNS = [
    r"^i need an? assessment\.?$",
    r"^what (do you|can you) (recommend|suggest|have)\??\s*$",
    r"^help me (find|choose|pick|select)\.?$",
    r"^assessment(s)?\s*$",
]


class Agent:
    """Stateless conversational agent for SHL assessment recommendation."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval = retrieval_engine
        self._catalog_map: dict[str, CatalogItem] = {}
        for item in retrieval_engine.get_all_items():
            self._catalog_map[item.name.lower().strip()] = item
            self._catalog_map[item.link] = item
        self._has_llm = bool(GROQ_API_KEY or GEMINI_API_KEY)

    async def process(self, messages: list[Message]) -> ChatResponse:
        """Process a conversation and return the next agent response."""
        try:
            # Step 1: Build retrieval query from conversation context
            search_query = self._build_search_query(messages)
            logger.info("Search query: %s", search_query[:200])

            # Step 2: Retrieve relevant assessments
            retrieved = self.retrieval.retrieve(search_query, top_k=TOP_K_RETRIEVAL)
            logger.info("Retrieved %d candidates", len(retrieved))

            # Step 3: Build catalog context for LLM
            catalog_context = self._format_catalog_context(retrieved)

            # Step 4: Call LLM (or heuristic fallback)
            if self._has_llm:
                raw_response = await self._call_llm(messages, catalog_context)
                logger.info("LLM raw response length: %d", len(raw_response))
                response = self._parse_response(raw_response)
            else:
                logger.info("No LLM API configured, using heuristic mode")
                response = self._heuristic_response(messages, retrieved)

            # Step 5: Validate recommendations against catalog
            response = self._validate_recommendations(response)

            return response

        except Exception as e:
            logger.exception("Agent processing error: %s", e)
            return ChatResponse(
                reply="I encountered an issue processing your request. Could you rephrase your question about SHL assessments?",
                recommendations=[],
                end_of_conversation=False,
            )

    # ---- Query Building ----

    def _build_search_query(self, messages: list[Message]) -> str:
        """Extract a retrieval query from conversation history.

        Combines all user messages with heavier weight on recent turns.
        Also extracts assessment names from assistant messages for refinement.
        """
        if not messages:
            return ""

        user_parts = []
        assistant_recs = []

        for msg in messages:
            if msg.role == "user":
                user_parts.append(msg.content)
            elif msg.role == "assistant":
                # Extract any assessment names the assistant previously mentioned
                # This helps with refinement/comparison scenarios
                for item_name in self._catalog_map:
                    if isinstance(self._catalog_map.get(item_name), CatalogItem):
                        if item_name in msg.content.lower():
                            assistant_recs.append(item_name)

        # Latest user message is most important (repeat it for emphasis)
        latest_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user = msg.content
                break

        query = f"{latest_user} {' '.join(user_parts)}"
        if assistant_recs:
            query += " " + " ".join(assistant_recs[:5])

        return query

    def _format_catalog_context(
        self, retrieved: list[tuple[CatalogItem, float]]
    ) -> str:
        """Format retrieved catalog items as context for the LLM."""
        if not retrieved:
            return "No relevant assessments found in catalog."

        lines = []
        for i, (item, score) in enumerate(retrieved, 1):
            lines.append(f"--- Assessment {i} (relevance: {score:.2f}) ---")
            lines.append(format_catalog_item_for_llm(item))
            lines.append("")

        return "\n".join(lines)

    # ---- LLM Calls ----

    async def _call_llm(self, messages: list[Message], catalog_context: str) -> str:
        """Call the LLM API. Tries Groq first, falls back to Gemini."""
        system_prompt = SYSTEM_PROMPT.format(catalog_context=catalog_context)

        llm_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            llm_messages.append({"role": msg.role, "content": msg.content})

        if GROQ_API_KEY:
            try:
                return await self._call_groq(llm_messages)
            except Exception as e:
                logger.warning("Groq call failed: %s", e)
                if GEMINI_API_KEY:
                    return await self._call_gemini(messages, system_prompt)
                raise

        if GEMINI_API_KEY:
            return await self._call_gemini(messages, system_prompt)

        raise RuntimeError("No LLM API key configured")

    async def _call_groq(self, messages: list[dict]) -> str:
        """Call Groq's OpenAI-compatible API."""
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _call_gemini(self, messages: list[Message], system_prompt: str) -> str:
        """Call Google Gemini API."""
        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "generationConfig": {
                        "temperature": LLM_TEMPERATURE,
                        "maxOutputTokens": LLM_MAX_TOKENS,
                        "responseMimeType": "application/json",
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    # ---- Heuristic Fallback ----

    def _heuristic_response(
        self,
        messages: list[Message],
        retrieved: list[tuple[CatalogItem, float]],
    ) -> ChatResponse:
        """Generate a response without an LLM, using rules and retrieval.

        Used as a fallback when no LLM API key is configured.
        """
        latest_user = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user = msg.content
                break

        latest_lower = latest_user.lower().strip()

        # Check for off-topic
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, latest_lower, re.IGNORECASE):
                return ChatResponse(
                    reply="I can only help with SHL assessment recommendations. That topic is outside my scope. What role are you hiring for?",
                    recommendations=[],
                    end_of_conversation=False,
                )

        # Check for confirmation (end of conversation)
        user_count = sum(1 for m in messages if m.role == "user")
        has_previous_recs = any(
            "recommend" in m.content.lower() or "|" in m.content
            for m in messages
            if m.role == "assistant"
        )
        for pattern in CONFIRMATION_PATTERNS:
            if re.search(pattern, latest_lower, re.IGNORECASE) and has_previous_recs:
                # Return the previous recommendations
                recs = self._extract_previous_recs(messages, retrieved)
                return ChatResponse(
                    reply="Shortlist confirmed. Let me know if you need anything else.",
                    recommendations=recs,
                    end_of_conversation=True,
                )

        # Check for vague query
        is_vague = any(
            re.search(p, latest_lower, re.IGNORECASE) for p in VAGUE_PATTERNS
        )
        is_first_turn = user_count <= 1

        # Check if query mentions specific technologies, skills, or roles
        # These signals indicate enough context even in short queries
        has_specific_signal = bool(
            re.search(
                r"\b(java|python|sql|docker|aws|angular|react|node|excel|word|"
                r"c\+\+|c#|\.net|ruby|php|scala|rust|go|swift|kotlin|spring|"
                r"accounting|finance|medical|hipaa|safety|leadership|sales|"
                r"customer\s*service|contact\s*cent|admin|engineer|developer|"
                r"analyst|manager|graduate|entry.level|senior|executive|"
                r"personality|cognitive|numerical|verbal|reasoning|"
                r"opq|verify|svar|situational|simulation)\b",
                latest_lower,
                re.IGNORECASE,
            )
        )

        if is_vague and not has_specific_signal:
            return ChatResponse(
                reply="I'd like to help you find the right SHL assessments. Could you tell me more about the role you're hiring for? Specifically, what function or domain (e.g., engineering, sales, customer service), and what seniority level?",
                recommendations=[],
                end_of_conversation=False,
            )

        if is_first_turn and len(latest_lower.split()) < 6 and not has_specific_signal:
            return ChatResponse(
                reply="I'd like to help you find the right SHL assessments. Could you tell me more about the role you're hiring for? Specifically, what function or domain (e.g., engineering, sales, customer service), and what seniority level?",
                recommendations=[],
                end_of_conversation=False,
            )

        # Enough context — recommend from retrieved items
        top_items = [item for item, score in retrieved[:8] if score > 0.05]
        if not top_items:
            return ChatResponse(
                reply="I couldn't find a strong match in the SHL catalog for that specific request. Could you describe the role or skills you need to assess?",
                recommendations=[],
                end_of_conversation=False,
            )

        recs = [
            Recommendation(name=item.name, url=item.link, test_type=item.test_type)
            for item in top_items[:7]
        ]

        reply = f"Based on your requirements, here are {len(recs)} recommended SHL assessments:"
        return ChatResponse(
            reply=reply,
            recommendations=recs,
            end_of_conversation=False,
        )

    def _extract_previous_recs(
        self,
        messages: list[Message],
        retrieved: list[tuple[CatalogItem, float]],
    ) -> list[Recommendation]:
        """Try to extract recommendations from previous assistant messages."""
        # Return top retrieved items as fallback
        top_items = [item for item, score in retrieved[:7] if score > 0.05]
        return [
            Recommendation(name=item.name, url=item.link, test_type=item.test_type)
            for item in top_items
        ]

    # ---- Response Parsing ----

    def _parse_response(self, raw: str) -> ChatResponse:
        """Parse LLM output into ChatResponse, handling malformed JSON."""
        # Try direct JSON parse
        try:
            parsed = json.loads(raw)
            return self._dict_to_response(parsed)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                return self._dict_to_response(parsed)
            except json.JSONDecodeError:
                pass

        # Try finding the outermost { ... } block
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        parsed = json.loads(raw[start : i + 1])
                        return self._dict_to_response(parsed)
                    except json.JSONDecodeError:
                        start = None

        logger.warning("Could not parse JSON from LLM response, using raw text")
        return ChatResponse(
            reply=raw[:500].strip(),
            recommendations=[],
            end_of_conversation=False,
        )

    def _dict_to_response(self, d: dict[str, Any]) -> ChatResponse:
        """Convert a parsed dict to ChatResponse."""
        reply = d.get("reply", d.get("response", d.get("message", "")))
        if not isinstance(reply, str):
            reply = str(reply)

        recs_raw = d.get("recommendations", [])
        recommendations = []
        if recs_raw and isinstance(recs_raw, list):
            for r in recs_raw:
                if isinstance(r, dict) and "name" in r:
                    recommendations.append(
                        Recommendation(
                            name=r.get("name", ""),
                            url=r.get("url", r.get("link", "")),
                            test_type=r.get("test_type", ""),
                        )
                    )

        end = d.get("end_of_conversation", False)
        if isinstance(end, str):
            end = end.lower() in ("true", "1", "yes")

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=bool(end),
        )

    # ---- Validation ----

    def _validate_recommendations(self, response: ChatResponse) -> ChatResponse:
        """Validate all recommendations come from the actual catalog.

        Fixes names/URLs/test_types to match catalog exactly.
        Drops hallucinated recommendations.
        """
        if not response.recommendations:
            return response

        validated: list[Recommendation] = []
        for rec in response.recommendations:
            item = self._find_catalog_item(rec.name, rec.url)
            if item:
                validated.append(
                    Recommendation(
                        name=item.name,
                        url=item.link,
                        test_type=item.test_type,
                    )
                )
            else:
                logger.warning(
                    "Dropping hallucinated recommendation: %s (%s)", rec.name, rec.url
                )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        deduped: list[Recommendation] = []
        for rec in validated:
            if rec.url not in seen_urls:
                seen_urls.add(rec.url)
                deduped.append(rec)

        # Enforce 1-10 limit
        if len(deduped) > 10:
            deduped = deduped[:10]

        response.recommendations = deduped
        return response

    def _find_catalog_item(self, name: str, url: str) -> CatalogItem | None:
        """Find a catalog item by name or URL with fuzzy matching."""
        # Try URL match first (most reliable)
        if url and url in self._catalog_map:
            item = self._catalog_map[url]
            if isinstance(item, CatalogItem):
                return item

        # Try exact name match
        name_lower = name.lower().strip()
        if name_lower in self._catalog_map:
            item = self._catalog_map[name_lower]
            if isinstance(item, CatalogItem):
                return item

        # Try retrieval engine's fuzzy name match
        item = self.retrieval.get_by_name(name)
        if item:
            return item

        # Try partial name matching across catalog
        best_match = None
        best_score = 0.0
        for stored_name, item in self._catalog_map.items():
            if not isinstance(item, CatalogItem):
                continue
            score = self._name_similarity(name_lower, stored_name)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = item

        return best_match

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Word-overlap Jaccard similarity between two names."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
