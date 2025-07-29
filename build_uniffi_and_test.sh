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
uniffi-bindgen generate src/tensor_iroh.udl --language python --out-dir .
# Create Python package structure
echo "Setting up Python package..."
mkdir -p tensor_iroh_py
cp tensor_iroh.py tensor_iroh_py/

# ------------------------------------------------------------------
# Copy + rename the compiled dynamic library so that UniFFI-Python
# can find it under the name it expects:  libuniffi_tensor_iroh.*
# ------------------------------------------------------------------
echo "Copying dynamic library for Python…"
case "$(uname -s)" in
    Linux*)
        cp target/release/libtensor_protocol.so \
           tensor_iroh_py/libuniffi_tensor_iroh.so
        ;;
    Darwin*)
        cp target/release/libtensor_protocol.dylib \
           tensor_iroh_py/libuniffi_tensor_iroh.dylib
        ;;
    MINGW*|MSYS*|CYGWIN*)
        cp target/release/tensor_protocol.dll \
           tensor_iroh_py/uniffi_tensor_iroh.dll
        ;;
    *)
        echo "Unsupported platform"; exit 1;;
esac

# Create __init__.py
cat > tensor_iroh_py/__init__.py << 'EOF'
"""
Tensor Iroh - Direct streaming tensor transfer over Iroh
"""
from .tensor_iroh import *

__version__ = "0.1.0"
EOF

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy

# Add the package to Python path for testing
export PYTHONPATH="$PWD/tensor_iroh_py:$PYTHONPATH"

echo "=== Build Complete ==="
echo "Files in tensor_iroh_py:"
ls -la tensor_iroh_py/
echo ""
echo "To test, run: python test_tensor_protocol_uniffi.py"
echo "Or run: PYTHONPATH=./tensor_iroh_py python test_tensor_protocol_uniffi.py" 