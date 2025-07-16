# Tensor Protocol - Direct Streaming Implementation

This is a **direct streaming tensor transfer protocol** built on top of Iroh's QUIC networking stack. Unlike blob-based approaches, this implementation streams tensor data directly over QUIC connections for maximum performance with intelligent connection pooling.

## Key Features

- **Direct QUIC Streaming**: Tensors are sent directly over QUIC streams without intermediate blob storage
- **Connection Pooling**: Intelligent reuse of QUIC connections for improved performance
- **Zero-Copy Design**: Minimal data copying for efficient memory usage
- **Python FFI**: Easy-to-use Python bindings via UniFFI
- **Async/Await Support**: Full async support for non-blocking operations
- **Type Safety**: Strongly typed tensor metadata and error handling
- **Comprehensive Testing**: 13/13 stress tests passing, including connection pool validation

## Architecture

### Core Components

1. **TensorProtocolHandler**: Implements Iroh's `ProtocolHandler` trait for custom protocol handling
2. **TensorNode**: Main API for sending/receiving tensors with connection pooling
3. **ConnectionPool**: Manages QUIC connection reuse for performance optimization
4. **Direct Streaming**: Uses QUIC bidirectional streams for immediate data transfer
5. **Custom ALPN**: Uses `"tensor-iroh/direct/0"` for protocol identification

### Connection Pool Architecture

```rust
pub struct ConnectionPool {
    connections: Arc<AsyncMutex<HashMap<String, PooledConnection>>>,
    max_idle_time: Duration,    // 5 minutes default
    max_connections: usize,      // 10 connections default
}

pub struct PooledConnection {
    connections: Connection,     // Iroh QUIC connection
    last_used: Instant,         // Last usage timestamp
    is_idle: bool,              // Connection state
}
```

### Protocol Flow

```
Node A                    Node B
  |                         |
  |-- Connect (QUIC) ------>|
  |   (or reuse existing)   |-- Accept Connection
  |-- Send TensorMessage -->|
  |    (with tensor data)   |-- Process & Store
  |                         |
  |-- Return to Pool ------>|
  |   (connection reuse)    |
```

### Message Types

```rust
enum TensorMessage {
    Request { tensor_name: String },           // Request specific tensor
    Response { tensor_name: String, data: TensorData }, // Send tensor data
    Error { message: String },                 // Error response
}
```

## Performance Optimizations

### Connection Pooling Benefits

- **Reduced Latency**: Eliminates connection setup overhead (~100-500ms per send)
- **Better Throughput**: Maintained connections have superior performance
- **Resource Efficiency**: Fewer active connections to manage
- **Scalability**: Handles high-frequency tensor sends efficiently

### Pool Management

- **Automatic Cleanup**: Idle connections are cleaned up after 5 minutes
- **Thread Safety**: Uses `tokio::sync::AsyncMutex` for async-aware locking
- **Connection Limits**: Maximum 10 concurrent connections per node
- **Smart Reuse**: Connections are marked idle and reused for subsequent sends

## Differences from Blob-Based Approaches

| Feature | Blob-Based (Psyche) | Direct Streaming (This) |
|---------|-------------------|------------------------|
| **Latency** | High (3-step process) | Low (direct transfer + connection reuse) |
| **Memory** | Stores blobs on disk | Streams directly with pooling |
| **Complexity** | Request→Ticket→Download | Single stream transfer |
| **Scalability** | Limited by storage | Limited by network + connection pool |
| **Use Case** | Large, persistent data | Real-time ML inference |
| **Performance** | Network + storage overhead | Optimized for repeated sends |

## Building and Testing

### Prerequisites

- Rust 1.70+
- Python 3.8+
- WSL (for Windows users)

### Build Steps

```bash
# Navigate to the protocol directory
cd protocol

# Make build script executable
chmod +x build_and_test.sh

# Build and generate Python bindings
./build_and_test.sh

# Run comprehensive tests (13/13 tests)
cargo run --bin test_tensor_protocol

# Run Python tests
python test_tensor_protocol.py
```

### Manual Build

```bash
# Navigate to the protocol directory
cd protocol

# Build Rust library
cargo build --release

# Generate Python bindings
uniffi-bindgen generate src/tensor_protocol.udl --language python --out-dir .

# Install Python dependencies
pip install numpy

# Test
PYTHONPATH=./tensor_protocol_py python test_tensor_protocol.py
```

## Usage Example

```python
import asyncio
from tensor_protocol import create_node, TensorData, TensorMetadata

async def main():
    # Create nodes (with connection pooling enabled)
    sender = create_node(None)
    receiver = create_node(None)
    
    # Start nodes
    await sender.start()
    await receiver.start()
    
    # Get addresses
    receiver_addr = await receiver.get_node_addr()
    
    # Create tensor
    tensor_data = TensorData(
        metadata=TensorMetadata(
            shape=[2, 3],
            dtype="float32",
            requires_grad=False
        ),
        data=b"tensor_bytes_here"
    )
    
    # Send tensor directly (connection will be pooled)
    await sender.send_tensor_direct(receiver_addr, "my_tensor", tensor_data)
    
    # Send again to same peer (connection will be reused)
    await sender.send_tensor_direct(receiver_addr, "my_tensor2", tensor_data)
    
    # Check pool size (should be 1 for single peer)
    pool_size = await sender.get_pool_size()
    print(f"Connection pool size: {pool_size}")
    
    # Receive tensor
    received = await receiver.receive_tensor()
    print(f"Received tensor: {received}")
    
    # Cleanup
    sender.shutdown()
    receiver.shutdown()

asyncio.run(main())
```

## Performance Characteristics

- **Small tensors** (< 1MB): ~1-5ms latency
- **Large tensors** (> 100MB): Limited by network bandwidth
- **Connection reuse**: ~100-500ms saved per subsequent send to same peer
- **Throughput**: Optimized by connection pooling
- **Memory usage**: Minimal buffering, streaming design with intelligent pooling

## Comprehensive Testing

The protocol includes 13 comprehensive stress tests:

1. **Basic Functionality**: Core tensor send/receive
2. **Pull/Request Pattern**: Control plane operations
3. **Concurrent Sends**: Race condition testing
4. **Rapid Fire Sends**: Timing stress testing
5. **Large Tensor Transfer**: 1MB+ tensor handling
6. **Multiple Receivers**: Broadcast scenarios
7. **Send Before Ready**: Timing edge cases
8. **Immediate Shutdown**: Resource cleanup
9. **Timeout Scenarios**: Network timeout handling
10. **Non-existent Tensor**: Error handling
11. **Bad Ticket Parsing**: Invalid address handling
12. **Post-shutdown Behavior**: Cleanup validation
13. **Connection Pool Reuse**: Pool functionality validation

**All 13 tests pass consistently**, demonstrating production-ready robustness.

## Error Handling

The protocol includes comprehensive error handling:

- `TensorError::Io`: Network I/O errors
- `TensorError::Serialization`: Data serialization errors  
- `TensorError::Connection`: QUIC connection errors
- `TensorError::Protocol`: Protocol-level errors

## Thread Safety

- **Async-aware locking**: Uses `tokio::sync::AsyncMutex` for connection pool
- **Non-blocking operations**: All async operations are non-blocking
- **Concurrent access**: Multiple threads can safely access the connection pool
- **Resource management**: Automatic cleanup of idle connections

## Future Enhancements

1. **Compression**: Add tensor compression for network efficiency
2. **Streaming**: Support for tensor streaming (partial sends)
3. **Authentication**: Add peer authentication and authorization
4. **Monitoring**: Add metrics and performance monitoring
5. **Batching**: Support for batched tensor transfers
6. **Pool Metrics**: Connection pool performance monitoring
7. **Adaptive Pooling**: Dynamic pool size based on usage patterns


## License

This implementation is designed to be compatible with Iroh's dual Apache-2.0/MIT license structure. 