#!/usr/bin/env bash
set -euo pipefail
cd /home/wzzz/LPRNet

echo '=== E9A ==='
bash scripts/train/run_green_e9a_exact_template_5prov.sh

echo '=== E9C ==='
bash scripts/train/run_green_e9c_exact_template_allprov_1800.sh

echo '=== E9B ==='
bash scripts/train/run_green_e9b_exact_template_allprov_11160.sh

echo '=== E9 all complete ==='
