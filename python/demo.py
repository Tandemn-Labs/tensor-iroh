import asyncio
import tensor_iroh as tp

async def main():
    # Create nodes (with connection pooling enabled)
    sender = tp.PyTensorNode()
    receiver = tp.PyTensorNode()
    
    # Start nodes
    await sender.start()
    await receiver.start()
    
    # Get addresses
    receiver_addr = await receiver.get_node_addr()
    
    # Create tensor data
    tensor_data = tp.PyTensorData(
        b"tensor_bytes_here",  # raw bytes
        [2, 3],               # shape
        "float32",            # dtype
        False                 # requires_grad
    )
    
    # Send tensor directly (connection will be pooled)
    await sender.send_tensor(receiver_addr, "my_tensor", tensor_data)
    
    # Send again to same peer (connection will be reused - much faster!)
    await sender.send_tensor(receiver_addr, "my_tensor2", tensor_data)
    
    # Check pool size (should be 1 for single peer)
    pool_size = await sender.pool_size()
    print(f"Connection pool size: {pool_size}")
    
    # Receive tensor
    received = await receiver.receive_tensor()
    if received:
        name, td = received  # receive_tensor() returns (name, PyTensorData)
        print(f"Received tensor '{name}' with shape: {td.shape}")
    
    # Receive second tensor
    received2 = await receiver.receive_tensor()
    if received2:
        name, td = received2  # receive_tensor() returns (name, PyTensorData)
        print(f"Received tensor '{name}' with shape: {td.shape}")
    
    # Cleanup
    sender.shutdown()
    receiver.shutdown()

asyncio.run(main())