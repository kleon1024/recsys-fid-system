#!/usr/bin/env bash
set -euo pipefail

job_name="${1:?job name is required}"
job_file="${HOME}/.config/recsys/jobs/${job_name}.env"

if [[ ! -f "${job_file}" ]]; then
  echo "missing job configuration: ${job_file}" >&2
  exit 2
fi

set -a
# The file is user-owned and intentionally defines RECSYS_COMMAND plus optional
# environment variables for one bounded systemd job.
source "${job_file}"
set +a

if [[ -z "${RECSYS_COMMAND:-}" ]]; then
  echo "RECSYS_COMMAND is required in ${job_file}" >&2
  exit 2
fi

mkdir -p "${HOME}/.local/state/recsys/jobs"
printf '%s start job=%s command=%s\n' \
  "$(date --iso-8601=seconds)" "${job_name}" "${RECSYS_COMMAND}"
exec bash -lc "${RECSYS_COMMAND}"
