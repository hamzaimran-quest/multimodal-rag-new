#!/bin/bash
set -euo pipefail

# PG 16 does not support transaction_timeout (added in PG 17).
sed '/^SET transaction_timeout/d' /seed/dvdrental.sql | psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB"
