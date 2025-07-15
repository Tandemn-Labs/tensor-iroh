#!/usr/bin/env python3
"""
Tensor Sender - Send tensors to a listening node
================================================

Usage: python tensor_sender.py <receiver_address>

Example: python tensor_sender.py node1abc123...
"""

import asyncio
import sys
import struct
import math
import random
import tensor_protocol as tp

def create_test_tensor(shape=(3, 4), random_data=True):
    """Create a test tensor with the given shape."""
    if random_data:
        # Create random float32 data
        n_elements = math.prod(shape)
        data = bytearray()
        for i in range(n_elements):
            val = random.uniform(-10.0, 10.0)
            data.extend(struct.pack("<f", val))
        return tp.PyTensorData(
            data=bytes(data),
            shape=list(shape),
            dtype="float32",
            requires_grad=False
        )
    else:
        # Create sequential data (0, 1, 2, ...)
        n_elements = math.prod(shape)
        data = bytearray()
        for i in range(n_elements):
            data.extend(struct.pack("<f", float(i)))
        return tp.PyTensorData(
            data=bytes(data),
            shape=list(shape),
            dtype="float32",
            requires_grad=False
        )

async def main():
    if len(sys.argv) != 2:
        print("Usage: python tensor_sender.py <receiver_address>")
        print("Example: python tensor_sender.py node1abc123...")
        sys.exit(1)
    
    receiver_addr = sys.argv[1]
    
    print("🚀 Starting Tensor Sender...")
    print(f"�� Connecting to: {receiver_addr}")
    
    # Create and start the sender node
    sender = tp.PyTensorNode()
    await sender.start()
    
    sender_addr = await sender.get_node_addr()
    print(f"📍 Sender address: {sender_addr}")
    
    try:
        # Create some test tensors
        tensors = [
            ("small_tensor", create_test_tensor(shape=(2, 3), random_data=False)),
            ("medium_tensor", create_test_tensor(shape=(10, 10), random_data=True)),
            ("large_tensor", create_test_tensor(shape=(50, 50), random_data=True)),
        ]
        
        for name, tensor in tensors:
            print(f"\n📦 Sending tensor '{name}' (shape: {tensor.shape})...")
            
            # Register the tensor
            sender.register_tensor(name, tensor)
            
            # Send the tensor
            await sender.send_tensor(receiver_addr, name, tensor)
            
            print(f"✅ Sent tensor '{name}' successfully!")
            
            # Small delay between sends
            await asyncio.sleep(1)
        
        print(f"\n🎉 All tensors sent successfully!")
        print("Press Ctrl+C to exit...")
        
        # Keep the node running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down sender...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        sender.shutdown()
        print("✅ Sender shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())