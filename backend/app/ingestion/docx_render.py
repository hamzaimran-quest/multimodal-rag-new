"""Render DOCX to PDF via LibreOffice headless (best-effort viewer artifact)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

_RENDER_TIMEOUT_SECONDS = 120


def _resolve_soffice() -> str | None:
    if settings.libreoffice_path:
        configured = Path(settings.libreoffice_path)
        if configured.is_file():
            return str(configured)

    found = shutil.which("soffice")
    if found:
        return found

    for candidate in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        path = Path(candidate)
        if path.is_file():
            return str(path)

    return None


def render_docx_to_pdf(docx_path: Path, output_path: Path) -> bool:
    """Render a DOCX file to a fixed viewer PDF path.

    Returns False when LibreOffice is unavailable or conversion fails. This is a
    soft dependency: DOCX ingestion must continue without a viewer PDF.
    """
    soffice = _resolve_soffice()
    if soffice is None:
        logger.warning(
            "LibreOffice (soffice) not found; skipping DOCX viewer PDF for %s. "
            "Install LibreOffice or set LIBREOFFICE_PATH to enable preview rendering.",
            docx_path.name,
        )
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(tmp_dir),
                    str(docx_path),
                ],
                capture_output=True,
                text=True,
                timeout=_RENDER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("LibreOffice timed out rendering %s", docx_path.name)
            return False
        except OSError as exc:
            logger.warning("LibreOffice failed to start for %s: %s", docx_path.name, exc)
            return False

        if result.returncode != 0:
            logger.warning(
                "LibreOffice render failed for %s (exit %s): %s",
                docx_path.name,
                result.returncode,
                (result.stderr or result.stdout or "").strip(),
            )
            return False

        produced = list(tmp_dir.glob("*.pdf"))
        if not produced:
            logger.warning("LibreOffice produced no PDF for %s", docx_path.name)
            return False

        shutil.move(str(produced[0]), str(output_path))
        logger.info("Rendered DOCX viewer PDF at %s", output_path)
        return True
