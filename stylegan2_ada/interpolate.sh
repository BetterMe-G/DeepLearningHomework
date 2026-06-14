#!/usr/bin/env bash
# Run interpolate.py on the latest snapshot.

set -euo pipefail
cd "$(dirname "$0")"

python3 interpolate.py --rows 5 --steps 16 --mode both --out_dir samples/
