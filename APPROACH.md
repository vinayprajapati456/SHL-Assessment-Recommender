# Approach Document — SHL Conversational Assessment Recommender

## Architecture

The system is a stateless FastAPI service with two endpoints: `GET /health` and `POST /chat`. Each `/chat` call receives the full conversation history, performs retrieval over the SHL catalog, constructs a grounded prompt, calls an LLM, and validates the output against the catalog before returning.

**Stack:** FastAPI · Groq (llama-3.3-70b-versatile) · scikit-learn TF-IDF · Pydantic · Docker → Render

## Retrieval Setup

The 377-item SHL catalog is loaded at startup. Each assessment is converted into a searchable document combining its name, description, test categories (Knowledge & Skills, Personality & Behavior, etc.), job levels, duration, and language support.

**Retrieval is a 6-phase pipeline:**

1. **Full-query TF-IDF** — cosine similarity over the entire user query against all assessment documents. Captures overall semantic relevance.
2. **Per-term TF-IDF** — each extracted technology/skill term (e.g., "java", "docker", "sql") is searched individually. This prevents multi-technology JDs from diluting any single term's signal.
3. **Keyword boosting** — exact keyword matches in assessment names/descriptions get score bonuses. Important for precise technology names.
4. **Name matching** — direct mentions of assessment names in the conversation get strong boosts.
5. **Core assessment injection** — OPQ32r and Verify G+ are always included in the candidate set, since they are relevant across most hiring scenarios.
6. **Metadata boosting** — optional filters for job level and test type.

This hybrid approach handles both specific queries ("need a Docker test") and broad queries ("hiring senior engineers") effectively. Retrieving 30 candidates gives the LLM enough to select from while staying within context limits.

**Why TF-IDF over embeddings:** For a 377-item catalog dominated by technology-specific terms, TF-IDF with bigrams outperforms dense retrieval on exact keyword matching (critical for "Java 8" vs "Java EE 7"). It loads in <100ms with zero GPU dependencies, critical for Render free tier cold starts and the 30-second timeout.

## Prompt Design

The system prompt encodes:

- **Role and scope constraints** — only discuss SHL assessments; refuse salary, legal, and off-topic questions
- **Decision logic** — when to clarify (vague queries lacking role/seniority/domain), when to recommend (2+ signals present or detailed JD provided), when to refine vs. start over
- **Output format** — strict JSON schema with `reply`, `recommendations` (array of `{name, url, test_type}`), and `end_of_conversation`
- **Full catalog context** — the top 30 retrieved assessments are injected verbatim with all metadata, so the LLM never needs to hallucinate

The LLM sees the real catalog data for every turn. Recommendations are post-validated: each name and URL is verified against the catalog. Any hallucinated item is silently dropped.

## Agent Design

The agent handles four behaviors:

- **Clarify:** First turn with <8 words or missing role/seniority → ask 1-2 targeted questions, return empty recommendations
- **Recommend:** Enough context → retrieve, select 1-10 assessments from candidates, explain the battery composition
- **Refine:** User adds/removes constraints → update the shortlist incrementally (the full conversation history provides context for what changed)
- **Compare:** User asks about differences → answer using only catalog description and metadata fields

Statelessness is maintained by reconstructing the search query from the full message history on every call. The latest user message is weighted most heavily, but all prior turns contribute context.

## Evaluation

**Metrics measured:**

1. **Recall@10** — fraction of expected assessments appearing in the final shortlist, averaged over 10 public conversation traces. The expected lists were extracted from the provided sample conversations.
2. **Schema compliance** — every response validated against the required JSON schema (reply string, recommendations array, end_of_conversation boolean).
3. **Behavioral probes** — binary assertions: refuses off-topic queries, doesn't recommend on vague first turn, honors mid-conversation edits, URLs always from catalog.

The evaluation script replays all 10 sample conversation scripts against the live endpoint, sending each user turn sequentially and collecting responses.

## What Didn't Work

- **Dense embeddings (sentence-transformers):** Added ~500MB memory footprint, didn't improve recall on technology-specific queries where exact keyword matching matters most. Removed in favor of lightweight TF-IDF.
- **Single-query TF-IDF:** When a JD mentions 7+ technologies, the combined query dilutes individual terms. Per-term search fixed this.
- **Unstructured LLM output:** Without `response_format: json_object`, the LLM occasionally wraps JSON in markdown fences or adds preamble text. JSON mode on Groq + robust fallback parsing solved this.

## AI Tools Used

Claude (Anthropic) was used for code scaffolding, prompt iteration, and evaluation script design. All architecture decisions, retrieval strategy, and prompt engineering reflect deliberate design choices that I can defend.
