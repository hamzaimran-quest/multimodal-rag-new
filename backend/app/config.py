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
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    images_dir: Path = Field(default=PROJECT_ROOT / "data" / "images", alias="IMAGES_DIR")
    uploads_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads", alias="UPLOADS_DIR")

    chunk_max_words: int = Field(default=400, alias="CHUNK_MAX_WORDS")
    chunk_overlap_words: int = Field(default=50, alias="CHUNK_OVERLAP_WORDS")

    chunks_index: str = "rag_chunks"
    documents_index: str = "rag_documents"
    hybrid_search_pipeline: str = "hybrid-search-pipeline"
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")

    # PDF side-panel viewer: how many pages to render before/after the cited page.
    pdf_viewer_page_window: int = Field(default=2, alias="PDF_VIEWER_PAGE_WINDOW")

    # DOCX viewer: pages to search forward from the previous match when locating text.
    docx_viewer_search_window_pages: int = Field(default=3, alias="DOCX_VIEWER_SEARCH_WINDOW_PAGES")
    # DOCX viewer: minimum fraction of a chunk's tokens that must be located in the
    # rendered PDF (in order) to accept a highlight. Below this we fail closed.
    docx_viewer_min_match_ratio: float = Field(default=0.6, alias="DOCX_VIEWER_MIN_MATCH_RATIO")

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
    # Intent gate (Track A): parallel Groq classifier for explicit "show me the image" queries.
    image_intent_enabled: bool = Field(default=True, alias="IMAGE_INTENT_ENABLED")
    image_intent_model: str = Field(default="llama-3.1-8b-instant", alias="IMAGE_INTENT_MODEL")
    # How many image chunks the intent-forced retrieval pass fetches.
    image_intent_top_k: int = Field(default=3, alias="IMAGE_INTENT_TOP_K")
    # How many intent images to actually display (explicit requests want the single
    # best match, not a montage).
    image_intent_max_display: int = Field(default=1, alias="IMAGE_INTENT_MAX_DISPLAY")
    # Relevance gate for intent images: keep only those scoring >= best * this ratio
    # (drops loosely-matching images like cover pages). Fail-closed if none qualify.
    image_intent_score_ratio: float = Field(default=0.6, alias="IMAGE_INTENT_SCORE_RATIO")
    # Absolute floor on the best intent image's score; below this, show no image.
    image_intent_min_score: float = Field(default=0.0, alias="IMAGE_INTENT_MIN_SCORE")
    # Recent chat turns fed to the answer LLM + query reformulation for follow-ups.
    chat_history_turns: int = Field(default=6, alias="CHAT_HISTORY_TURNS")
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


settings = Settings()
