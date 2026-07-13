"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_JWT_SECRETS = frozenset(
    {
        "",
        "dev-insecure-change-me",
        "change-me",
        "your_jwt_secret_here",
    }
)

# Project root is two levels up from backend/app/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    opensearch_host: str = Field(default="localhost", alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, alias="OPENSEARCH_PORT")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_answer_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_ANSWER_MODEL")
    groq_aux_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_AUX_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    images_dir: Path = Field(default=PROJECT_ROOT / "data" / "images", alias="IMAGES_DIR")
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads", alias="UPLOADS_DIR")

    chunk_max_words: int = Field(default=400, alias="CHUNK_MAX_WORDS")
    chunk_overlap_words: int = Field(default=50, alias="CHUNK_OVERLAP_WORDS")

    chunks_index: str = "rag_chunks"
    documents_index: str = "rag_documents"
    hybrid_search_pipeline: str = "hybrid-search-pipeline"
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")

    # PDF side-panel viewer: how many pages to render before/after the cited page.
    pdf_viewer_page_window: int = Field(default=2, alias="PDF_VIEWER_PAGE_WINDOW")

    # DOCX viewer: pages to search forward from the previous match when locating text.
    docx_viewer_search_window_pages: int = Field(default=3, alias="DOCX_VIEWER_SEARCH_WINDOW_PAGES")
    # DOCX viewer: minimum fraction of a chunk's tokens that must be located in the
    # rendered PDF (in order) to accept a highlight. Below this we fail closed.
    docx_viewer_min_match_ratio: float = Field(default=0.6, alias="DOCX_VIEWER_MIN_MATCH_RATIO")

    # XLSX ingestion: row band sizes when a sheet has no Excel Table objects.
    excel_row_band_size: int = Field(default=30, alias="EXCEL_ROW_BAND_SIZE")
    excel_row_band_size_medium: int = Field(default=15, alias="EXCEL_ROW_BAND_SIZE_MEDIUM")
    excel_row_band_size_wide: int = Field(default=10, alias="EXCEL_ROW_BAND_SIZE_WIDE")
    excel_medium_column_threshold: int = Field(default=6, alias="EXCEL_MEDIUM_COLUMN_THRESHOLD")
    excel_wide_column_threshold: int = Field(default=10, alias="EXCEL_WIDE_COLUMN_THRESHOLD")
    excel_top_k: int = Field(default=5, alias="EXCEL_TOP_K")
    pdf_top_k: int = Field(default=7, alias="PDF_TOP_K")
    pdf_table_slots: int = Field(default=3, alias="PDF_TABLE_SLOTS")
    pdf_table_candidate_pool: int = Field(default=8, alias="PDF_TABLE_CANDIDATE_POOL")

    # XLSX workbook schema: LLM proposes joins, code validates and enriches at ingestion.
    excel_schema_enabled: bool = Field(default=True, alias="EXCEL_SCHEMA_ENABLED")
    excel_schema_model: str = Field(default="openai/gpt-oss-20b", alias="EXCEL_SCHEMA_MODEL")
    excel_schema_sample_rows: int = Field(default=20, alias="EXCEL_SCHEMA_SAMPLE_ROWS")
    excel_schema_min_overlap_ratio: float = Field(
        default=0.9, alias="EXCEL_SCHEMA_MIN_OVERLAP_RATIO"
    )
    excel_schema_soft_link_overlap_ratio: float = Field(
        default=0.85, alias="EXCEL_SCHEMA_SOFT_LINK_OVERLAP_RATIO"
    )
    excel_schema_timeout_seconds: float = Field(default=45.0, alias="EXCEL_SCHEMA_TIMEOUT_SECONDS")
    excel_schema_log_max_chars: int = Field(default=4000, alias="EXCEL_SCHEMA_LOG_MAX_CHARS")
    excel_schema_max_retries: int = Field(default=3, alias="EXCEL_SCHEMA_MAX_RETRIES")
    excel_anchor_max_entities: int = Field(default=3, alias="EXCEL_ANCHOR_MAX_ENTITIES")
    excel_cluster_expand_per_anchor: int = Field(default=12, alias="EXCEL_CLUSTER_EXPAND_PER_ANCHOR")

    # Image attachment: surface relevant images beside text answers (PDF only).
    image_attach_enabled: bool = Field(default=True, alias="IMAGE_ATTACH_ENABLED")
    # Proximity (Track B): consider at most N top text/table hits as attachment anchors.
    image_proximity_anchor_count: int = Field(default=3, alias="IMAGE_PROXIMITY_ANCHOR_COUNT")
    # Only treat a hit as an anchor if its score >= top_score * this ratio (score cliff guard).
    image_proximity_score_ratio: float = Field(default=0.7, alias="IMAGE_PROXIMITY_SCORE_RATIO")
    # Minimum horizontal (column) overlap ratio required before considering vertical proximity.
    image_proximity_column_overlap_min: float = Field(
        default=0.25, alias="IMAGE_PROXIMITY_COLUMN_OVERLAP_MIN"
    )
    # Vertical gap (PDF points) beyond which an image is considered unrelated to the anchor.
    image_proximity_margin_px: float = Field(default=80.0, alias="IMAGE_PROXIMITY_MARGIN_PX")
    # Minimum combined attachment score [0,1] to attach an image. Fail closed below this.
    image_min_attachment_score: float = Field(default=0.3, alias="IMAGE_MIN_ATTACHMENT_SCORE")
    # Max images shown in the hero strip per answer (dedup + cap).
    image_max_display: int = Field(default=2, alias="IMAGE_MAX_DISPLAY")
    # Bbox IoU above which two images on the same page are treated as duplicates.
    image_dedup_iou: float = Field(default=0.6, alias="IMAGE_DEDUP_IOU")
    # How many image chunks the agent search_images tool retrieves.
    image_intent_top_k: int = Field(default=3, alias="IMAGE_INTENT_TOP_K")
    # DOCX image proximity: attach images within ±N body blocks of a text/table anchor.
    docx_image_proximity_block_radius: int = Field(
        default=2, alias="DOCX_IMAGE_PROXIMITY_BLOCK_RADIUS"
    )

    # SQL Agent (optional user PostgreSQL connections)
    sql_agent_enabled: bool = Field(default=True, alias="SQL_AGENT_ENABLED")
    sql_agent_model: str | None = Field(default=None, alias="SQL_AGENT_MODEL")
    sql_agent_max_steps: int = Field(default=10, alias="SQL_AGENT_MAX_STEPS")
    sql_agent_query_timeout_seconds: int = Field(default=30, alias="SQL_AGENT_QUERY_TIMEOUT_SECONDS")
    sql_agent_max_rows: int = Field(default=100, alias="SQL_AGENT_MAX_ROWS")
    sql_credentials_key: str | None = Field(default=None, alias="SQL_CREDENTIALS_KEY")
    sql_connection_cache_ttl_seconds: int = Field(default=300, alias="SQL_CONNECTION_CACHE_TTL_SECONDS")
    sql_schema_max_tables: int = Field(default=40, alias="SQL_SCHEMA_MAX_TABLES")
    sql_schema_max_chars: int = Field(default=12000, alias="SQL_SCHEMA_MAX_CHARS")
    sql_scope_classifier_enabled: bool = Field(default=True, alias="SQL_SCOPE_CLASSIFIER_ENABLED")
    sql_scope_classifier_model: str | None = Field(default=None, alias="SQL_SCOPE_CLASSIFIER_MODEL")
    sql_scope_classifier_timeout_seconds: float = Field(default=20.0, alias="SQL_SCOPE_CLASSIFIER_TIMEOUT_SECONDS")
    sql_scope_min_confidence: float = Field(default=0.55, alias="SQL_SCOPE_MIN_CONFIDENCE")
    # Optional explicit path to LibreOffice soffice binary (soft dependency for DOCX preview).
    libreoffice_path: str | None = Field(default=None, alias="LIBREOFFICE_PATH")

    # Auth + persistence
    database_url: str = Field(
        default="postgresql+psycopg2://rag:rag@localhost:5432/rag",
        alias="DATABASE_URL",
    )
    jwt_secret: str = Field(default="dev-insecure-change-me", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_ttl_minutes: int = Field(default=15, alias="ACCESS_TOKEN_TTL_MINUTES")
    refresh_token_ttl_days: int = Field(default=7, alias="REFRESH_TOKEN_TTL_DAYS")
    refresh_cookie_name: str = Field(default="rag_refresh", alias="REFRESH_COOKIE_NAME")
    cookie_secure: bool = Field(default=False, alias="COOKIE_SECURE")
    cookie_samesite: str = Field(default="lax", alias="COOKIE_SAMESITE")

    # Rate limiting (per user_id, requests per minute; 0 = disabled)
    upload_rate_limit_per_minute: int = Field(default=10, alias="UPLOAD_RATE_LIMIT_PER_MINUTE")
    query_rate_limit_per_minute: int = Field(default=30, alias="QUERY_RATE_LIMIT_PER_MINUTE")

    # Agent router: Groq tool-calling for retrieval routing
    agent_model: str = Field(default="openai/gpt-oss-120b", alias="AGENT_MODEL")
    agent_max_rounds: int = Field(default=1, alias="AGENT_MAX_ROUNDS")
    # Short snippet per chunk in search_documents tool JSON (router only; full text used for answers).
    agent_tool_snippet_max_chars: int = Field(default=200, alias="AGENT_TOOL_SNIPPET_MAX_CHARS")

    # Follow-up handling: aux model rewrites using prior user queries only (no answers/chunks).
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")
    query_rewrite_model: str = Field(default="openai/gpt-oss-20b", alias="QUERY_REWRITE_MODEL")
    chat_history_turns: int = Field(default=6, alias="CHAT_HISTORY_TURNS")
    # Max chars per prior user question included in rewrite context.
    chat_history_query_max_chars: int = Field(default=400, alias="CHAT_HISTORY_QUERY_MAX_CHARS")
    # Max chars of the most recent assistant reply passed into the next query (rewrite + answer).
    chat_last_reply_max_chars: int = Field(default=800, alias="CHAT_LAST_REPLY_MAX_CHARS")
    # Max chars of retrieved context passed to the grounded answer LLM (0 = no limit).
    llm_context_max_chars: int = Field(default=12000, alias="LLM_CONTEXT_MAX_CHARS")

    # Chart tool: aux LLM builds Chart.js config; QuickChart renders the image URL.
    chart_llm_model: str = Field(default="openai/gpt-oss-20b", alias="CHART_LLM_MODEL")
    chart_table_max_chars: int = Field(default=4000, alias="CHART_TABLE_MAX_CHARS")
    quickchart_width: int = Field(default=500, alias="QUICKCHART_WIDTH")
    quickchart_height: int = Field(default=300, alias="QUICKCHART_HEIGHT")

    # Comma-separated allowed browser origins for CORS (credentials required for refresh cookie).
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        alias="CORS_ORIGINS",
    )

    # When true, refuse startup if JWT_SECRET is missing or a known insecure default.
    require_secure_jwt_secret: bool = Field(default=False, alias="REQUIRE_SECURE_JWT_SECRET")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _strip_cors_origins(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @property
    def cors_origin_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def jwt_secret_is_secure(self) -> bool:
        return self.jwt_secret.strip() not in _INSECURE_JWT_SECRETS

    def validate_production_secrets(self) -> None:
        if self.require_secure_jwt_secret and not self.jwt_secret_is_secure:
            raise RuntimeError(
                "JWT_SECRET is missing or uses an insecure default. "
                "Set a strong random secret in the environment (REQUIRE_SECURE_JWT_SECRET=true)."
            )

    @property
    def refresh_token_ttl_seconds(self) -> int:
        return self.refresh_token_ttl_days * 24 * 60 * 60

    @property
    def opensearch_url(self) -> str:
        return f"http://{self.opensearch_host}:{self.opensearch_port}"

    @property
    def resolved_images_dir(self) -> Path:
        return self._resolve_path(self.images_dir)

    @property
    def resolved_uploads_dir(self) -> Path:
        return self._resolve_path(self.uploads_dir)

    def _resolve_path(self, path: Path) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")

    @property
    def resolved_sql_agent_model(self) -> str:
        return self.sql_agent_model or self.groq_aux_model

    @property
    def resolved_sql_scope_classifier_model(self) -> str:
        return self.sql_scope_classifier_model or self.groq_aux_model


settings = Settings()
