#!/usr/bin/env bash
set -Eeuo pipefail

# Backwards-compatible alias.  The old implementation used the seven-task
# Table-2 list and could silently omit GSM8K/SVAMP/SciQ/WinoGrande.  Keep the
# historical filename callable, but make the requested nine-task launcher the
# single source of truth for model order, output isolation, and resume logic.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/01_generate_all_models.sh" "$@"
