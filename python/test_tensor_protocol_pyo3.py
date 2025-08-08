#!/usr/bin/env python3
"""
Tensor-Iroh integration test-bench (PyO3 edition)
=====================================================

* Uses the `tensor_iroh` PyO3 module you just compiled.
* Exercises the same scenarios as the old UniFFI test-bench:
  ◦ basic send/receive
  ◦ large-tensor path
  ◦ node-address sanity
  ◦ optional Torch round-trip
"""

import asyncio, struct, random, math, logging
from typing import Optional, Tuple, List

# ──────────── import PyO3 bindings ────────────
import tensor_iroh as tp          # the Rust module
TensorNode      = tp.PyTensorNode    # alias for brevity
PyTensorData    = tp.PyTensorData

# ───────────── optional Torch support ─────────
try:
    import torch
    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False
    torch = None            # type: ignore

# ───────────────── helper utils ───────────────
def tensor_bytes(shape: Tuple[int, ...]) -> bytes:
    """Deterministic float32 ramp – pure Python."""
    n = math.prod(shape)
    ba = bytearray()
    for i in range(n):
        ba.extend(struct.pack("<f", float(i)))
    return bytes(ba)


def make_td(shape=(3, 4), *, dtype="float32", randomish=False) -> PyTensorData:
    """Create a PyTensorData from raw bytes."""
    if dtype != "float32":
        raise ValueError("Only float32 supported")
    if randomish:
        ba = bytearray()
        for i in range(math.prod(shape)):
            val = float(i) + random.random()
            ba.extend(struct.pack("<f", val))
        raw = bytes(ba)
    else:
        raw = tensor_bytes(shape)

    return PyTensorData(raw, list(shape), dtype, False)


def td_equal(a: PyTensorData, b: PyTensorData) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.as_bytes() == b.as_bytes()
    )


# -------- Torch helpers --------
def torch_to_td(t: "torch.Tensor") -> PyTensorData:
    if t.dtype != torch.float32:
        raise ValueError("only float32 tensors supported")
    if not t.is_contiguous():
        t = t.contiguous()
    return PyTensorData(
        t.cpu().numpy().tobytes(),
        list(t.shape),
        "float32",
        bool(t.requires_grad),
    )


def td_to_torch(td: PyTensorData) -> "torch.Tensor":
    import numpy as np

    arr = (
        np.frombuffer(td.as_bytes(), dtype=np.float32)
        .reshape(td.shape)
        .copy()  # own the buffer
    )
    return torch.from_numpy(arr)


# ─────────────── smoke tests ────────────────
async def smoke_direct() -> bool:
    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()

    addr2 = await n2.get_node_addr()
    td = make_td()

    n1.register_tensor("t", td)
    await n1.send_tensor(addr2, "t", td)

    name, data = await n2.wait_for_tensor()
    ok = td_equal(td, data)
    n1.shutdown(); n2.shutdown()
    return ok


async def smoke_large() -> bool:
    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    td = make_td(shape=(64, 64), randomish=True)
    n1.register_tensor("big", td)
    await n1.send_tensor(addr2, "big", td)

    name, data = await n2.wait_for_tensor()
    ok = td_equal(td, data)
    n1.shutdown(); n2.shutdown()
    return ok


async def smoke_address() -> bool:
    n = TensorNode()
    await n.start()
    addr = await n.get_node_addr()
    n.shutdown()
    return addr.startswith("node")


async def test_multiple_64kb_tensors() -> bool:
    """Test sending 20 64KB tensors repeatedly."""
    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    # Create 20 different 64KB tensors (16384 float32 values = 64KB)
    tensors = []
    for i in range(20):
        # Create a 128x128 float32 tensor (16384 elements = 64KB)
        td = make_td(shape=(128, 128), randomish=True)
        tensors.append((f"tensor_{i}", td))

    # Send all 20 tensors
    for name, td in tensors:
        await n1.send_tensor(addr2, name, td)

    # Receive all 20 tensors and verify
    received_tensors = {}
    for _ in range(1000):  # Give plenty of time for all tensors
        try:
            name, data = await asyncio.wait_for(n2.wait_for_tensor(), timeout=1.0)
            received_tensors[name] = data
            if len(received_tensors) == 20:
                break
        except asyncio.TimeoutError:
            break

    # Verify we received all 20 tensors with correct data
    if len(received_tensors) != 20:
        print(f"❌ Expected 20 tensors, received {len(received_tensors)}")
        n1.shutdown(); n2.shutdown()
        return False

    # Verify each tensor matches what we sent
    for original_name, original_td in tensors:
        if original_name not in received_tensors:
            print(f"❌ Missing tensor: {original_name}")
            n1.shutdown(); n2.shutdown()
            return False
        
        received_td = received_tensors[original_name]
        if not td_equal(original_td, received_td):
            print(f"❌ Tensor data mismatch for: {original_name}")
            n1.shutdown(); n2.shutdown()
            return False

    print(f"✅ Successfully sent and received all 20 64KB tensors")
    n1.shutdown(); n2.shutdown()
    return True


# ───────────── Torch tests (optional) ─────────────
async def torch_direct() -> bool:
    if not _HAS_TORCH:
        print("ℹ️  torch not installed – skipping")
        return True

    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    td = torch_to_td(t)
    n1.register_tensor("torch", td)
    await n1.send_tensor(addr2, "torch", td)

    name, data = await n2.wait_for_tensor()
    same = torch.allclose(t, td_to_torch(data))
    n1.shutdown(); n2.shutdown()
    return same


async def torch_large() -> bool:
    if not _HAS_TORCH:
        print("ℹ️  torch not installed – skipping")
        return True

    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    t = torch.randn(128, 128, dtype=torch.float32)
    td = torch_to_td(t)
    n1.register_tensor("torch_big", td)
    await n1.send_tensor(addr2, "torch_big", td)

    name, data = await n2.wait_for_tensor()
    same = torch.allclose(t, td_to_torch(data))
    n1.shutdown(); n2.shutdown()
    return same


async def test_tensor_name_return() -> bool:
    """Test that tensor names are properly returned when receiving tensors."""
    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    # Create test tensor
    td = make_td(shape=(2, 3), randomish=True)
    
    # Send tensor with specific name
    test_name = "my_test_tensor_123"
    await n1.send_tensor(addr2, test_name, td)

    # Receive tensor (no polling)
    name, data = await n2.wait_for_tensor()
    if name == test_name and td_equal(td, data):
        n1.shutdown(); n2.shutdown()
        return True
    else:
        print(f"❌ Name mismatch: expected '{test_name}', got '{name}'")
        n1.shutdown(); n2.shutdown()
        return False


# ────────────── test harness ──────────────
async def main() -> None:
    tests = [
        ("Node addressing", smoke_address),
        ("Direct bytes tensor", smoke_direct),
        ("Large bytes tensor", smoke_large),
        ("Multiple 64KB tensors", test_multiple_64kb_tensors),
        ("Direct torch tensor", torch_direct),
        ("Large torch tensor", torch_large),
        ("Tensor name return", test_tensor_name_return),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n━━ {name} ━━")
        ok = await fn()
        print("✅ PASS" if ok else "❌ FAIL")
        passed += int(ok)

    print(f"\n=== {passed}/{len(tests)} tests passed ===")
    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
