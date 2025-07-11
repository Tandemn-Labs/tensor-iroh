use tensor_protocol::{create_node, TensorData, TensorMetadata};
use tokio::time::{sleep, Duration};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Rust Test: Tensor Protocol Direct Streaming ===");

    // Create two TensorNode instances
    let node1 = create_node(None);
    let node2 = create_node(None);

    // Start both nodes
    println!("Starting nodes...");
    node1.start().await?;
    node2.start().await?;

    // Obtain their addresses
    let addr1 = node1.get_node_addr().await?;
    let addr2 = node2.get_node_addr().await?;
    println!("Node1 address: {}", addr1);
    println!("Node2 address: {}", addr2);

    // Construct a test tensor: shape [3,4], float32, deterministic data
    let shape = vec![3, 4];
    let dtype = "float32".to_string();
    let requires_grad = false;
    let metadata = TensorMetadata { shape: shape.clone(), dtype: dtype.clone(), requires_grad };
    let num_elems = (shape[0] * shape[1]) as usize;
    let mut bytes = Vec::with_capacity(num_elems * 4);
    for i in 0..num_elems {
        let v = i as f32;
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let tensor = TensorData { metadata: metadata.clone(), data: bytes.clone() };
    println!("Created test tensor of {} elements", num_elems);

    // Register the tensor on node1
    println!("Registering tensor on node1...");
    node1.register_tensor("test_tensor".to_string(), tensor.clone())?;

    // Send the tensor from node1 to node2
    println!("Sending tensor from node1 to node2...");
    node1.send_tensor_direct(addr2.clone(), "test_tensor".to_string(), tensor.clone()).await?;

    // Wait briefly for delivery
    sleep(Duration::from_millis(100)).await;

    // Attempt to receive the tensor on node2
    println!("Receiving tensor on node2...");
    let mut received = None;
    for attempt in 1..=50 {
        if let Some(tx) = node2.receive_tensor().await? {
            received = Some(tx);
            println!("✅ Received on attempt {}", attempt);
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }

    // Validate the result
    let received = received.expect("Failed to receive tensor");
    // Compare metadata fields individually since TensorMetadata doesn't implement PartialEq
    assert_eq!(received.metadata.shape, metadata.shape);
    assert_eq!(received.metadata.dtype, metadata.dtype);
    assert_eq!(received.metadata.requires_grad, metadata.requires_grad);
    assert_eq!(received.data, bytes);
    println!("✅ SUCCESS: Tensor metadata and data match!");

    // Clean up
    node1.shutdown()?;
    node2.shutdown()?;
    println!("Shut down nodes.");
    Ok(())
} 