"""Security guardrails for chat, SQL, and grounded answers."""

from app.security.input_classifier import InputVerdict, classify_user_input, classify_user_input_async
from app.security.output_guard import OutputVerdict, scan_output_text
from app.security.sql_policy import (
    SqlPolicyError,
    reject_dangerous_functions,
    reject_ddl_sql,
    reject_system_catalog_access,
    validate_sql_allowed,
)

__all__ = [
    "InputVerdict",
    "OutputVerdict",
    "SqlPolicyError",
    "classify_user_input",
    "classify_user_input_async",
    "reject_dangerous_functions",
    "reject_ddl_sql",
    "reject_system_catalog_access",
    "scan_output_text",
    "validate_sql_allowed",
]
