"""Central configuration for the service.

Every tunable in the system lives in this one class so that a reviewer can see, in a
single screen, which knobs exist and what their defaults are.  Values are read from
environment variables and from a local ``.env`` file via pydantic-settings, which also
validates types at import time.  A typo such as ``RETRIEVAL_TOP_K=five`` therefore
fails immediately at startup with a clear message instead of deep inside a request.

Field names map to environment variables case-insensitively, so ``llm_model`` is set
with ``LLM_MODEL=...``.  See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/src/raapid_rag/config.py  ->  parents[2] == <repo>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
# Prebuilt indexes live here, one directory per corpus fingerprint.
DEFAULT_CACHE_DIR: Path = PROJECT_ROOT / "data" / ".index"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated variables in .env must not break startup
    )

    # ------------------------------------------------------------------ LLM
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key. Optional so that retrieval-only mode and tests run without it.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Chat model. A small model is the default on purpose: the grounding "
        "design must hold up without relying on a frontier model's judgement.",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_max_output_tokens: int = Field(default=1200, ge=64)

    # ------------------------------------------------------ chunking / index
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model. 384-dim, max 256 word-piece tokens.",
    )
    chunk_target_tokens: int = Field(
        default=220,
        ge=50,
        le=256,
        description="Upper bound on tokens per chunk. Capped at 256 because the embedding "
        "model silently truncates anything longer, which would make chunk tails unsearchable.",
    )
    chunk_overlap_tokens: int = Field(default=40, ge=0)

    # -------------------------------------------------------------- retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_max_chunks_per_doc: int = Field(
        default=3,
        ge=1,
        description="Diversity cap. Without it a long, on-topic document (a guideline) fills every "
        "top-k slot and a multi-document question never sees the second document. Observed in "
        "the Step 4 live run: the HCC-mapping question retrieved four guideline chunks and no "
        "Assessment chunk from the patient's note.",
    )
    retrieval_min_dense_score: float = Field(
        default=0.37,
        ge=-1.0,
        le=1.0,
        description="Refusal gate, dense half. Best cosine similarity between the query and "
        "any retrieved chunk. Absolute, not rank-based: RRF scores cannot gate because RRF "
        "always ranks *something* first. Provisional until calibrated in the eval step.",
    )
    retrieval_min_bm25_score: float = Field(
        default=5.0,
        ge=0.0,
        description="Refusal gate, lexical half. Best raw BM25 score. Lets an exact identifier "
        "match (an ICD code, a policy number) pass even when the dense score is weak. The gate "
        "refuses only when BOTH halves are below threshold. Provisional.",
    )
    retrieval_enable_document_anchoring: bool = Field(
        default=True,
        description="When the query contains an identifier that belongs to one document (a "
        "patient's name, member id, patient ref, doc id, or policy/guideline number), pull that "
        "document in as a block before filling the remaining slots by rank. Only identifiers "
        "anchor, never topical words. Mirrors how a coder works: open the chart, then read the "
        "section. Fixes the multi-hop recall failure observed in the Step 4 live run.",
    )
    retrieval_anchor_doc_token_budget: int = Field(
        default=900,
        ge=1,
        description="An anchored document shorter than this is included whole (in document order); "
        "a longer one contributes only its best-ranked chunks, up to the per-document cap.",
    )
    retrieval_max_anchored_docs: int = Field(default=2, ge=1)
    retrieval_context_token_budget: int = Field(
        default=1800,
        ge=1,
        description="Hard cap on retrieved-context tokens sent to the model. Chunks beyond "
        "the budget are dropped lowest-score-first. This is the first line of defence "
        "against context-window overflow.",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant used to merge BM25 and dense rankings.",
    )

    # ------------------------------------------------------------- agent loop
    max_tool_rounds: int = Field(default=4, ge=0, le=10)
    tool_timeout_seconds: float = Field(default=5.0, gt=0)
    max_repair_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="How many times the validation error is fed back to the model before "
        "the service gives up and returns a deterministic error envelope.",
    )
    prompt_version: Literal["v1", "v2", "v3", "v4"] = Field(
        default="v4",
        description="Which system prompt to load. Earlier versions are kept so the iteration can be "
        "demonstrated; see llm/prompts/PROMPT_CHANGELOG.md.",
    )
    fault_injection: Literal["", "bad_citation", "malformed_json"] = Field(
        default="",
        description="Deliberately corrupt the first model response so the repair path can "
        "be shown live. Must be empty in any real deployment.",
    )

    # ------------------------------------------------------------- guardrails
    max_query_chars: int = Field(default=2000, ge=1)
    phi_policy: Literal["flag", "block"] = Field(
        default="flag",
        description="What to do when the query contains PHI-like patterns (SSN, phone, DOB, email). "
        "'flag' adds a warning (mock data); 'block' returns an error envelope (real data).",
    )
    retrieval_strong_dense_score: float = Field(
        default=0.65,
        description="Dense score at which retrieval strength counts as 1.0 in the confidence composite "
        "(relevant probes on this corpus reached 0.65 to 0.72).",
    )
    retrieval_strong_bm25_score: float = Field(default=15.0, description="BM25 score counted as full strength.")
    confidence_high: float = Field(default=0.80, ge=0.0, le=1.0)
    confidence_medium: float = Field(default=0.55, ge=0.0, le=1.0)
    tool_policy: dict[str, str] = Field(
        default={
            "icd10_lookup": r"\bhcc\b|\bicd[- ]?10\b|\bmap(?:s|ped)?\s+to\b",
            "check_eligibility": r"\beligib|\bcoverage\b|\bsubmitt?(?:ed|able)\b|\bbe submitted\b",
        },
        description="Tool name -> regex over the question. If the pattern matches and the tool was not "
        "called successfully, the answer is flagged for review and capped below the high band. Found "
        "by the Step 6 eval: a softened prompt answered an HCC question at confidence 1.00 without the "
        "ICD-10 lookup. The prompt asks for the tool; this makes the omission visible.",
    )

    tool_refusal_topics: dict[str, list[str]] = Field(
        default={"check_eligibility": ["coverage", "eligib", "inactive", "terminat", "submit"]},
        description="Tool name -> words that mark a refusal as being *about* that tool's topic. When the "
        "model refuses, its stated reason mentions one of these words, the tool ran, and the question "
        "did not match the tool's policy pattern, the refusal is treated as tool-induced and repaired "
        "once. Found by the Step 6 eval: a coding question refused because coverage was inactive.",
    )

    # ---------------------------------------------------------------- service
    llm_max_concurrency: int = Field(
        default=8,
        ge=1,
        description="Per-process cap on simultaneous model calls. Protects the provider rate limit "
        "when many users arrive at once; more capacity means more processes, not a higher number.",
    )
    answer_cache_size: int = Field(
        default=256,
        ge=0,
        description="LRU cache of answers keyed by normalised question, scope and index fingerprint. "
        "Repeated policy questions from many users cost nothing after the first. 0 disables.",
    )

    # ------------------------------------------------------------------ paths
    corpus_dir: Path = Field(default=PROJECT_ROOT / "data" / "corpus")
    prompts_dir: Path = Field(default=PROJECT_ROOT / "src" / "raapid_rag" / "llm" / "prompts")

    # ------------------------------------------------- observability / PHI
    log_level: str = Field(default="INFO")
    log_chunk_text: bool = Field(
        default=False,
        description="Never log raw chunk text by default. With real clinical data, logs are "
        "the most common place PHI leaks.",
    )

    @property
    def llm_available(self) -> bool:
        """True when an API key is configured; used to fall back to retrieval-only mode."""
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached after the first call)."""
    return Settings()
