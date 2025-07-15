#!/usr/bin/env python3
"""
Tensor Receiver - Listen for incoming tensors
=============================================

Usage: python tensor_receiver.py

This script starts a node and listens for incoming tensors,
displaying information about each received tensor.
"""

import asyncio
import struct
import numpy as np
import tensor_protocol as tp

def analyze_tensor(tensor):
    """Analyze and display information about a received tensor."""
    print(f"\n📊 Tensor Analysis:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Requires grad: {tensor.requires_grad}")
    print(f"   Data size: {len(tensor.as_bytes())} bytes")
    
    # Convert to numpy for analysis
    data = np.frombuffer(tensor.as_bytes(), dtype=np.float32)
    data = data.reshape(tensor.shape)
    
    print(f"   Min value: {data.min():.4f}")
    print(f"   Max value: {data.max():.4f}")
    print(f"   Mean value: {data.mean():.4f}")
    print(f"   Std dev: {data.std():.4f}")
    
    # Show first few values
    if data.size <= 10:
        print(f"   Values: {data.flatten()}")
    else:
        print(f"   First 5 values: {data.flatten()[:5]}")
        print(f"   Last 5 values: {data.flatten()[-5:]}")

async def main():
    print("🎧 Starting Tensor Receiver...")
    
    # Create and start the receiver node
    receiver = tp.PyTensorNode()
    await receiver.start()
    
    receiver_addr = await receiver.get_node_addr()
    print(f"📍 Receiver address: {receiver_addr}")
    print(f"📡 Waiting for incoming tensors...")
    print(f"�� Share this address with the sender:")
    print(f"   {receiver_addr}")
    print(f"\nPress Ctrl+C to stop listening...")
    
    tensor_count = 0
    
    try:
        while True:
            # Wait for incoming tensors
            received = await receiver.receive_tensor()
            
            if received:
                tensor_count += 1
                print(f"\n�� Received tensor #{tensor_count}!")
                analyze_tensor(received)
                
                # Optional: Try to convert to PyTorch if available
                try:
                    if hasattr(received, 'to_torch'):
                        torch_tensor = received.to_torch()
                        print(f"�� Converted to PyTorch tensor: {torch_tensor.shape}")
                except Exception as e:
                    print(f"⚠️  PyTorch conversion failed: {e}")
            else:
                # No tensor received, continue listening
                await asyncio.sleep(0.1)
                
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down receiver...")
        print(f"📈 Total tensors received: {tensor_count}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        receiver.shutdown()
        print("✅ Receiver shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())