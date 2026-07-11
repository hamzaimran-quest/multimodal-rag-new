"""LLM-proposed workbook schema with deterministic code validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

from app.config import settings
from app.ingestion.xlsx_workbook import (
    WorkbookData,
    build_workbook_summary_for_llm,
    headers_match,
    normalize_header,
    resolve_header_index,
)
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL

logger = logging.getLogger(__name__)

Cardinality = Literal["one_to_one", "one_to_many"]
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_SYSTEM_PROMPT = """You analyze Excel workbooks for relational structure between sheets.

Each sheet has row 1 as column headers and row 2+ as data.

Output ONLY one JSON object (no markdown fences):

{
  "clusters": [
    {
      "primary_sheet": "sheet name of the main entity table",
      "primary_key_column": "column on the primary sheet that identifies each row",
      "satellites": [
        {
          "sheet": "satellite sheet name",
          "key_column": "column on the satellite that joins to the primary key",
          "payload_columns": ["col1", "col2"],
          "cardinality": "one_to_one" or "one_to_many"
        }
      ]
    }
  ],
  "standalone_sheets": ["sheets with no validated join to any cluster"]
}

Rules:
- A workbook may have MULTIPLE independent clusters (e.g. Orders+OrderLines AND Products+Reviews).
- Do not join clusters that share no key relationship.
- payload_columns lists attribute columns to enrich from the satellite (exclude the key column).
- For a single-sheet workbook, return clusters: [] and standalone_sheets: [that sheet].
- cardinality one_to_one: at most one satellite row per key; one_to_many: multiple satellite rows per key.
- Use exact header strings from the input.
- If unsure about a join, omit it rather than guess."""


@dataclass
class ValidatedSatellite:
    sheet_name: str
    key_column: str
    key_col_index: int
    payload_columns: list[str]
    payload_col_indices: list[int]
    cardinality: Cardinality
    overlap_ratio: float
    key_repeat_ratio: float


@dataclass
class ValidatedCluster:
    primary_sheet: str
    primary_key_column: str
    primary_key_col_index: int
    satellites: list[ValidatedSatellite] = field(default_factory=list)


@dataclass
class SchemaValidationEntry:
    cluster_primary: str | None
    satellite_sheet: str | None
    status: Literal["accepted", "rejected"]
    reason: str
    overlap_ratio: float | None = None
    key_repeat_ratio: float | None = None


@dataclass
class ValidatedWorkbookSchema:
    clusters: list[ValidatedCluster] = field(default_factory=list)
    standalone_sheets: list[str] = field(default_factory=list)
    llm_proposal: dict[str, Any] | None = None
    llm_error: str | None = None
    validation_log: list[SchemaValidationEntry] = field(default_factory=list)

    def to_document_metadata(self) -> dict[str, Any]:
        return {
            "clusters": [
                {
                    "primary_sheet": cluster.primary_sheet,
                    "primary_key_column": cluster.primary_key_column,
                    "satellites": [
                        {
                            "sheet": satellite.sheet_name,
                            "key_column": satellite.key_column,
                            "payload_columns": satellite.payload_columns,
                            "cardinality": satellite.cardinality,
                            "overlap_ratio": round(satellite.overlap_ratio, 4),
                            "key_repeat_ratio": round(satellite.key_repeat_ratio, 4),
                        }
                        for satellite in cluster.satellites
                    ],
                }
                for cluster in self.clusters
            ],
            "standalone_sheets": list(self.standalone_sheets),
            "validation_log": [asdict(entry) for entry in self.validation_log],
            "llm_error": self.llm_error,
        }


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_llm_proposal(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "not_object"
    clusters = parsed.get("clusters")
    if clusters is not None and not isinstance(clusters, list):
        return None, "clusters_not_list"
    standalone = parsed.get("standalone_sheets")
    if standalone is not None and not isinstance(standalone, list):
        return None, "standalone_not_list"
    return parsed, None


def propose_workbook_schema(workbook: WorkbookData) -> tuple[dict[str, Any] | None, str | None]:
    """One LLM call per workbook: propose clusters and join keys."""
    if not settings.excel_schema_enabled:
        return None, "disabled"
    if not settings.groq_configured:
        return None, "groq_not_configured"

    summary = build_workbook_summary_for_llm(
        workbook,
        sample_row_limit=settings.excel_schema_sample_rows,
    )
    user_content = json.dumps(summary, ensure_ascii=False, default=str)
    payload = {
        "model": settings.excel_schema_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.excel_schema_timeout_seconds) as client:
            response = client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "XLSX_SCHEMA_LLM request_failed status=%s model=%s user_content_chars=%s body=%s",
                    response.status_code,
                    settings.excel_schema_model,
                    len(user_content),
                    response.text[: settings.excel_schema_log_max_chars],
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
    except httpx.HTTPStatusError as exc:
        response = exc.response
        logger.warning(
            "XLSX_SCHEMA_LLM http_error status=%s model=%s error=%s",
            response.status_code if response is not None else None,
            settings.excel_schema_model,
            exc,
            exc_info=True,
        )
        return None, f"request_failed:{exc}"
    except Exception as exc:
        logger.warning("XLSX_SCHEMA_LLM request_failed error=%s", exc, exc_info=True)
        return None, f"request_failed:{exc}"

    proposal, error = _parse_llm_proposal(content)
    logger.info(
        "XLSX_SCHEMA_LLM model=%s sheets=%s proposal_chars=%s error=%s raw=%s",
        settings.excel_schema_model,
        len(workbook.sheets),
        len(content),
        error,
        content[: settings.excel_schema_log_max_chars],
    )
    return proposal, error


def _key_repeat_ratio(values: list[str]) -> float:
    non_empty = [value for value in values if value]
    if not non_empty:
        return 0.0
    unique = len(set(non_empty))
    return 1.0 - (unique / len(non_empty))


def _overlap_ratio(satellite_keys: list[str], primary_keys: set[str]) -> float:
    non_empty = [value for value in satellite_keys if value]
    if not non_empty:
        return 0.0
    matched = sum(1 for value in non_empty if value in primary_keys)
    return matched / len(non_empty)


def _detect_cardinality(satellite_keys: list[str]) -> Cardinality:
    repeat_ratio = _key_repeat_ratio(satellite_keys)
    if repeat_ratio > 0.01:
        return "one_to_many"
    return "one_to_one"


def _validate_satellite(
    workbook: WorkbookData,
    *,
    primary_sheet: SheetData,
    primary_key_column: str,
    primary_keys: set[str],
    satellite_spec: dict[str, Any],
) -> tuple[ValidatedSatellite | None, SchemaValidationEntry]:
    sheet_name = str(satellite_spec.get("sheet") or "").strip()
    key_column = str(satellite_spec.get("key_column") or "").strip()
    payload_columns_raw = satellite_spec.get("payload_columns") or []
    proposed_cardinality = str(satellite_spec.get("cardinality") or "").strip().lower()

    satellite = workbook.sheet_by_name.get(sheet_name)
    if satellite is None:
        return None, SchemaValidationEntry(
            cluster_primary=primary_sheet.name,
            satellite_sheet=sheet_name or None,
            status="rejected",
            reason="satellite_sheet_not_found",
        )

    key_index = resolve_header_index(satellite.headers, key_column)
    if key_index is None:
        return None, SchemaValidationEntry(
            cluster_primary=primary_sheet.name,
            satellite_sheet=sheet_name,
            status="rejected",
            reason="key_column_not_found",
        )

    payload_columns: list[str] = []
    payload_indices: list[int] = []
    for raw_name in payload_columns_raw:
        name = str(raw_name).strip()
        if not name or headers_match(name, key_column):
            continue
        col_index = resolve_header_index(satellite.headers, name)
        if col_index is None:
            return None, SchemaValidationEntry(
                cluster_primary=primary_sheet.name,
                satellite_sheet=sheet_name,
                status="rejected",
                reason=f"payload_column_not_found:{name}",
            )
        payload_columns.append(satellite.headers[col_index])
        payload_indices.append(col_index)

    if not payload_columns:
        all_payload_indices = [
            index
            for index, header in enumerate(satellite.headers)
            if index != key_index and normalize_header(header)
        ]
        payload_columns = [satellite.headers[index] for index in all_payload_indices]
        payload_indices = all_payload_indices

    satellite_keys = satellite.column_values(key_column)
    overlap = _overlap_ratio(satellite_keys, primary_keys)
    repeat_ratio = _key_repeat_ratio(satellite_keys)
    detected = _detect_cardinality(satellite_keys)

    if overlap < settings.excel_schema_min_overlap_ratio:
        return None, SchemaValidationEntry(
            cluster_primary=primary_sheet.name,
            satellite_sheet=sheet_name,
            status="rejected",
            reason="overlap_below_threshold",
            overlap_ratio=overlap,
            key_repeat_ratio=repeat_ratio,
        )

    cardinality: Cardinality = detected
    if proposed_cardinality in {"one_to_one", "one_to_many"}:
        cardinality = proposed_cardinality  # type: ignore[assignment]

    return (
        ValidatedSatellite(
            sheet_name=sheet_name,
            key_column=satellite.headers[key_index],
            key_col_index=key_index,
            payload_columns=payload_columns,
            payload_col_indices=payload_indices,
            cardinality=cardinality,
            overlap_ratio=overlap,
            key_repeat_ratio=repeat_ratio,
        ),
        SchemaValidationEntry(
            cluster_primary=primary_sheet.name,
            satellite_sheet=sheet_name,
            status="accepted",
            reason="ok",
            overlap_ratio=overlap,
            key_repeat_ratio=repeat_ratio,
        ),
    )


def validate_workbook_schema(
    workbook: WorkbookData,
    proposal: dict[str, Any] | None,
    *,
    llm_error: str | None = None,
) -> ValidatedWorkbookSchema:
    """Validate LLM proposal against full workbook columns; fail closed per join."""
    result = ValidatedWorkbookSchema(
        llm_proposal=proposal,
        llm_error=llm_error,
    )
    all_sheet_names = {sheet.name for sheet in workbook.sheets}
    assigned_sheets: set[str] = set()

    if not proposal:
        result.standalone_sheets = sorted(all_sheet_names)
        logger.info(
            "XLSX_SCHEMA_VALIDATE outcome=standalone_only sheets=%s llm_error=%s",
            len(result.standalone_sheets),
            llm_error,
        )
        return result

    for cluster_spec in proposal.get("clusters") or []:
        if not isinstance(cluster_spec, dict):
            continue
        primary_name = str(cluster_spec.get("primary_sheet") or "").strip()
        primary_key_column = str(cluster_spec.get("primary_key_column") or "").strip()
        primary_sheet = workbook.sheet_by_name.get(primary_name)
        if primary_sheet is None:
            result.validation_log.append(
                SchemaValidationEntry(
                    cluster_primary=primary_name or None,
                    satellite_sheet=None,
                    status="rejected",
                    reason="primary_sheet_not_found",
                )
            )
            continue

        primary_key_index = resolve_header_index(primary_sheet.headers, primary_key_column)
        if primary_key_index is None:
            result.validation_log.append(
                SchemaValidationEntry(
                    cluster_primary=primary_name,
                    satellite_sheet=None,
                    status="rejected",
                    reason="primary_key_column_not_found",
                )
            )
            continue

        primary_keys = {
            value
            for value in primary_sheet.column_values(primary_sheet.headers[primary_key_index])
            if value
        }
        if not primary_keys:
            result.validation_log.append(
                SchemaValidationEntry(
                    cluster_primary=primary_name,
                    satellite_sheet=None,
                    status="rejected",
                    reason="primary_key_empty",
                )
            )
            continue

        validated_satellites: list[ValidatedSatellite] = []
        for satellite_spec in cluster_spec.get("satellites") or []:
            if not isinstance(satellite_spec, dict):
                continue
            satellite, entry = _validate_satellite(
                workbook,
                primary_sheet=primary_sheet,
                primary_key_column=primary_sheet.headers[primary_key_index],
                primary_keys=primary_keys,
                satellite_spec=satellite_spec,
            )
            result.validation_log.append(entry)
            if satellite is not None:
                validated_satellites.append(satellite)
                assigned_sheets.add(satellite.sheet_name)

        if validated_satellites:
            result.clusters.append(
                ValidatedCluster(
                    primary_sheet=primary_name,
                    primary_key_column=primary_sheet.headers[primary_key_index],
                    primary_key_col_index=primary_key_index,
                    satellites=validated_satellites,
                )
            )
            assigned_sheets.add(primary_name)
        else:
            result.standalone_sheets.append(primary_name)

    standalone_set = set(result.standalone_sheets)
    for sheet_name in sorted(all_sheet_names):
        if sheet_name in assigned_sheets or sheet_name in standalone_set:
            continue
        standalone_set.add(sheet_name)
    result.standalone_sheets = sorted(standalone_set)

    logger.info(
        "XLSX_SCHEMA_VALIDATE clusters=%s standalone=%s accepted_joins=%s rejected_joins=%s detail=%s",
        len(result.clusters),
        len(result.standalone_sheets),
        sum(1 for entry in result.validation_log if entry.status == "accepted"),
        sum(1 for entry in result.validation_log if entry.status == "rejected"),
        json.dumps([asdict(entry) for entry in result.validation_log], ensure_ascii=False)[:2000],
    )
    return result


def detect_and_validate_workbook_schema(workbook: WorkbookData) -> ValidatedWorkbookSchema:
    proposal, llm_error = propose_workbook_schema(workbook)
    return validate_workbook_schema(workbook, proposal, llm_error=llm_error)
