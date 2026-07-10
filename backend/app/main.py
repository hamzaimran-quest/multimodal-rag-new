"""Application entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chats import router as chats_router
from app.api.documents import router as documents_router
from app.api.debug import router as debug_router
from app.api.health import router as health_router
from app.api.images import router as images_router
from app.api.pdfs import router as pdfs_router
from app.api.spreadsheet import router as spreadsheet_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.config import settings
from app.db.session import init_db
from app.opensearch.bootstrap import bootstrap_opensearch, wait_for_opensearch
from app.opensearch.client import get_opensearch_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
app.include_router(debug_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(images_router)
app.include_router(pdfs_router)
app.include_router(spreadsheet_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "multimodal-rag",
        "phase": "5",
        "docs": "/docs",
    }
