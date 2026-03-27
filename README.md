# tensor-iroh

Direct peer-to-peer tensor streaming over QUIC. No intermediate storage, no disk I/O -- just memory-to-memory transfers between machines.

Built on [Iroh](https://iroh.computer/)'s QUIC networking stack with connection pooling, automatic chunking for large payloads, and relay fallback for NAT traversal.

```
pip install tensor_iroh
```

## Quick Start

```python
import asyncio
import tensor_iroh as tp

async def main():
    sender = tp.PyTensorNode()
    receiver = tp.PyTensorNode()

    await sender.start()
    await receiver.start()

    receiver_addr = await receiver.get_node_addr()

    tensor = tp.PyTensorData(
        b"raw_tensor_bytes",   # raw bytes (or NumPy array)
        [2, 3],                # shape
        "float32",             # dtype
        False                  # requires_grad
    )

    # First send establishes a QUIC connection (~150-300ms)
    await sender.send_tensor(receiver_addr, "weights", tensor)

    # Subsequent sends reuse the pooled connection (~1-5ms)
    await sender.send_tensor(receiver_addr, "gradients", tensor)

    # Async wait for incoming tensors (no polling)
    name, data = await receiver.wait_for_tensor()
    print(f"Got '{name}', shape={data.shape}, dtype={data.dtype}")

    sender.shutdown()
    receiver.shutdown()

asyncio.run(main())
```

## Why This Is Fast

| | Traditional (request-store-download) | tensor-iroh |
|---|---|---|
| **Steps** | 3 (request, write to disk, download) | 1 (direct stream) |
| **Repeat sends** | Full overhead each time | Pooled connection reuse |
| **Latency (small tensor)** | 100-500ms | 1-5ms (pooled) |
| **Large tensors** | Limited by storage I/O | Streams in 1MB chunks, up to 10GB |
| **Memory** | Intermediate disk buffers | Memory-to-memory streaming |

## Features

- **Direct QUIC streaming** -- tensors go straight from sender to receiver, no blob storage
- **Connection pooling** -- connections stay warm for 30 days, up to 50 per node
- **Chunked transfers** -- payloads over 1MB are automatically chunked; supports up to 10GB
- **Async/await** -- `wait_for_tensor()` yields on arrival, no busy-polling
- **TLS 1.3** -- all traffic encrypted by default via QUIC
- **NAT traversal** -- automatic relay fallback when direct connections aren't possible
- **Zero-copy** -- NumPy arrays and bytes pass through without extra copies
- **PyTorch support** -- optional `torch` feature for direct Torch tensor integration

## Architecture

```
Node A (sender)                      Node B (receiver)
  |                                    |
  |-- Get/create connection from pool  |
  |-- Open bidirectional stream ------>|-- Accept stream
  |-- Send TensorMessage (chunked) -->|-- Reassemble & deserialize
  |-- Return connection to pool        |-- Deliver via async channel
  |                                    |
  |   (next send reuses connection)    |-- await wait_for_tensor()
```

The protocol uses a custom ALPN (`tensor-iroh/direct/0`) and [postcard](https://docs.rs/postcard) binary serialization. Messages are length-prefixed for payloads under 1MB and chunk-prefixed for larger ones.

### Core Types

```rust
// What gets sent over the wire
enum TensorMessage {
    Request  { tensor_name: String },
    Response { tensor_name: String, data: TensorData },
    Error    { message: String },
}

// Tensor payload
struct TensorData {
    metadata: TensorMetadata,  // shape, dtype, requires_grad
    data: Vec<u8>,             // raw bytes
}
```

## Python API

### PyTensorNode

| Method | Description |
|---|---|
| `start()` | Initialize the QUIC endpoint and start listening |
| `get_node_addr()` | Get this node's address (share with peers) |
| `send_tensor(addr, name, data)` | Send a tensor to a peer (connection-pooled) |
| `wait_for_tensor()` | Async wait for the next incoming tensor |
| `receive_tensor()` | Non-blocking poll for a received tensor |
| `register_tensor(name, data)` | Store a tensor locally for pull requests |
| `pool_size()` | Number of active pooled connections |
| `shutdown()` | Clean shutdown |

### PyTensorData

```python
td = tp.PyTensorData(data, shape, dtype, requires_grad)
td.shape          # list[int]
td.dtype          # str
td.requires_grad  # bool
td.data_bytes     # bytes
td.to_numpy()     # numpy.ndarray
```

## Building from Source

### Prerequisites

- Rust 1.70+
- Python 3.8+
- [maturin](https://github.com/PyO3/maturin) (`pip install maturin`)

### PyO3 Bindings (recommended)

```bash
# Build and install the wheel
maturin build --release -F python --out ./target/wheels
pip install ./target/wheels/tensor_iroh-*.whl

# Or with Torch support (longer compile)
maturin build --release -F "python,torch" --out ./target/wheels
```

There's also a convenience script:

```bash
chmod +x build_pyo3_bindings.sh
./build_pyo3_bindings.sh
```

### UniFFI Bindings (alternative)

UniFFI bindings generate a cross-language Python package via `tensor_iroh.udl`:

```bash
chmod +x build_uniffi_and_test.sh
./build_uniffi_and_test.sh
```

### Rust Library Only

```bash
cargo build --release
```

## Testing

### Rust Stress Tests

The project includes 13 stress tests covering core functionality, concurrency, large transfers, error handling, and connection pool behavior:

```bash
cargo run --release --bin test_tensor_protocol
```

### Python Tests

```bash
python python/test_tensor_protocol_pyo3.py     # PyO3 bindings
python python/test_tensor_protocol_uniffi.py    # UniFFI bindings
```

## Distributed Inference Demo

The `monkey_patched_code/` directory contains a distributed inference demo using a FastAPI central server that orchestrates tensor exchange across peer nodes:

```bash
# Terminal 1: Start the central server
python monkey_patched_code/central_server_demo.py

# Terminal 2+: Start peer nodes
python monkey_patched_code/peer_node_demo.py

# Or run everything together
python monkey_patched_code/run_demo.py
```

## Configuration

These constants are defined in `src/lib.rs`:

| Constant | Default | Description |
|---|---|---|
| `MAX_CONNECTIONS` | 50 | Max pooled connections per node |
| `CONNECTION_IDLE_TIMEOUT` | 30 days | How long idle connections stay in the pool |
| `CHUNK_SIZE` | 1 MB | Chunk size for large message transfer |
| `MAX_MESSAGE_SIZE` | 10 GB | Maximum single tensor transfer size |
| `TENSOR_ALPN` | `tensor-iroh/direct/0` | Protocol identifier |

## Error Handling

```rust
enum TensorError {
    Io { message: String },              // Network/file I/O
    Serialization { message: String },   // postcard encode/decode
    Connection { message: String },      // QUIC connection issues
    Protocol { message: String },        // Protocol-level errors
}
```

## License

Apache-2.0 / MIT dual license, compatible with Iroh's licensing.
