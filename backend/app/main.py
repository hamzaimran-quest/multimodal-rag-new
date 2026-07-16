"""Application entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.chats import router as chats_router
from app.api.documents import router as documents_router
from app.api.debug import router as debug_router
from app.api.health import router as health_router
from app.api.images import router as images_router
from app.api.pdfs import router as pdfs_router
from app.api.sql_agent import router as sql_agent_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.api.spreadsheet import router as spreadsheet_router
from app.config import settings
from app.db.session import init_db
from app.ingestion.embeddings import start_embedding_model_warmup
from app.opensearch.bootstrap import bootstrap_opensearch, wait_for_opensearch
from app.opensearch.client import get_opensearch_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vite build output copied to /app/static in the Docker image (outside bind-mounted app/).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

images_dir = Path(settings.resolved_images_dir)
images_dir.mkdir(parents=True, exist_ok=True)
Path(settings.resolved_uploads_dir).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_production_secrets()
    init_db()
    logger.info("Database tables ensured")

    client = wait_for_opensearch()
    result = bootstrap_opensearch(client)
    logger.info("OpenSearch bootstrap complete: %s", result)

    app.state.opensearch = client

    start_embedding_model_warmup()
    logger.info("Embedding model warm-up started in background")

    yield
    client.close()


app = FastAPI(
    title="Multimodal RAG API",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Range"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(documents_router)
if settings.debug_endpoints_enabled:
    app.include_router(debug_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(images_router)
app.include_router(pdfs_router)
app.include_router(spreadsheet_router)
app.include_router(sql_agent_router)


def _mount_spa() -> None:
    """Serve the built React app from /app/static when present (Docker image)."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        @app.get("/")
        async def root() -> dict[str, str]:
            return {
                "service": "multimodal-rag",
                "phase": "5",
                "docs": "/docs",
            }

        logger.info("SPA static assets not found at %s — API-only mode", STATIC_DIR)
        return

    assets_dir = STATIC_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def spa_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        candidate = (STATIC_DIR / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)

    logger.info("SPA static assets mounted from %s", STATIC_DIR)


_mount_spa()
