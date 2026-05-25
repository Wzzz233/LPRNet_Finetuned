#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.conda"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  echo "Linux conda environment not found or incomplete: ${ENV_PATH}" >&2
  exit 1
fi

echo "Use one of the following:"
echo "  conda activate ${ENV_PATH}"
echo "  conda run -p ${ENV_PATH} python train_LPRNet.py --help"
