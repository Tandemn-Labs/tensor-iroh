#!/usr/bin/env bash
# build_bindings.sh – build UniFFI stubs AND a PyO3 wheel
set -euo pipefail

root_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$root_dir"

echo "=== (1) Clean the crate ==="
cargo clean

echo "=== (2) Build PyO3 wheel via maturin (feature: python) ==="
if ! command -v maturin &> /dev/null; then
  pip install maturin
fi
# Build *in place* (editable-like): puts .so into ./target/wheels/ and installs it.
maturin develop --release -F python

echo "✅ PyO3 wheel built & installed in your active venv."
echo " Try in Python:"
echo "     PYTHONPATH=./tensor_protocol_py:$PYTHONPATH"
echo "     import tensor_protocol"

echo "=== (3) Done ==="
