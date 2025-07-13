#!/bin/bash
set -e

echo "=== Building Tensor Protocol ==="
echo "Building for WSL environment..."

# Make sure we're in the protocol directory
cd "$(dirname "$0")"

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean

# Build the Rust library
echo "Building Rust library..."
cargo build --release

# Install uniffi-bindgen if not present
if ! command -v uniffi-bindgen &> /dev/null; then
    echo "Installing uniffi-bindgen..."
    # Corrected install command
    cargo install --git https://github.com/mozilla/uniffi-rs --locked uniffi_bindgen
fi

# Generate Python bindings
echo "Generating Python bindings..."
uniffi-bindgen generate src/tensor_protocol.udl --language python --out-dir .
# Create Python package structure
echo "Setting up Python package..."
mkdir -p tensor_protocol_py
cp tensor_protocol.py tensor_protocol_py/

# ------------------------------------------------------------------
# Copy + rename the compiled dynamic library so that UniFFI-Python
# can find it under the name it expects:  libuniffi_tensor_protocol.*
# ------------------------------------------------------------------
echo "Copying dynamic library for Python…"
case "$(uname -s)" in
    Linux*)
        cp target/release/libtensor_protocol.so \
           tensor_protocol_py/libuniffi_tensor_protocol.so
        ;;
    Darwin*)
        cp target/release/libtensor_protocol.dylib \
           tensor_protocol_py/libuniffi_tensor_protocol.dylib
        ;;
    MINGW*|MSYS*|CYGWIN*)
        cp target/release/tensor_protocol.dll \
           tensor_protocol_py/uniffi_tensor_protocol.dll
        ;;
    *)
        echo "Unsupported platform"; exit 1;;
esac

# Create __init__.py
cat > tensor_protocol_py/__init__.py << 'EOF'
"""
Tensor Protocol - Direct streaming tensor transfer over Iroh
"""
from .tensor_protocol import *

__version__ = "0.1.0"
EOF

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy

# Add the package to Python path for testing
export PYTHONPATH="$PWD/tensor_protocol_py:$PYTHONPATH"

echo "=== Build Complete ==="
echo "Files in tensor_protocol_py:"
ls -la tensor_protocol_py/
echo ""
echo "To test, run: python test_tensor_protocol.py"
echo "Or run: PYTHONPATH=./tensor_protocol_py python test_tensor_protocol.py" 