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

# Copy Python bindings
cp tensor_protocol.py tensor_protocol_py/

# Copy the compiled library based on platform
# Try each platform's library naming convention and copy to BOTH locations
if [ -f "target/release/libtensor_protocol.so" ]; then
    echo "Found Linux shared library"
    # Copy to tensor_protocol_py/ for package use
    cp target/release/libtensor_protocol.so tensor_protocol_py/libuniffi_tensor_protocol.so
    # Copy to main directory for direct testing
    cp target/release/libtensor_protocol.so libuniffi_tensor_protocol.so
elif [ -f "target/release/libtensor_protocol.dylib" ]; then
    echo "Found macOS shared library"
    # Copy to tensor_protocol_py/ for package use
    cp target/release/libtensor_protocol.dylib tensor_protocol_py/libuniffi_tensor_protocol.so
    # Copy to main directory for direct testing
    cp target/release/libtensor_protocol.dylib libuniffi_tensor_protocol.so
elif [ -f "target/release/tensor_protocol.dll" ]; then
    echo "Found Windows DLL"
    # Copy to tensor_protocol_py/ for package use
    cp target/release/tensor_protocol.dll tensor_protocol_py/libuniffi_tensor_protocol.so
    # Copy to main directory for direct testing
    cp target/release/tensor_protocol.dll libuniffi_tensor_protocol.so
else
    echo "Error: Could not find compiled library in target/release/"
    echo "Please ensure the build completed successfully"
    exit 1
fi

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
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=== Build Complete ==="
echo "To test, run: python test_tensor_protocol.py"
echo "Or run: PYTHONPATH=$PWD python test_tensor_protocol.py" 