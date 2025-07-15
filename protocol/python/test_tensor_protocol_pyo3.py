#!/usr/bin/env python3
"""
Tensor-Protocol integration test-bench (PyO3 edition)
=====================================================

* Uses the `tensor_protocol` PyO3 module you just compiled.
* Exercises the same scenarios as the old UniFFI test-bench:
  ◦ basic send/receive
  ◦ large-tensor path
  ◦ node-address sanity
  ◦ optional Torch round-trip
"""

import asyncio, struct, random, math, logging
from typing import Optional, Tuple, List

# ──────────── import PyO3 bindings ────────────
import tensor_protocol as tp          # the Rust module
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

    for _ in range(50):
        if (r := await n2.receive_tensor()):
            ok = td_equal(td, r)
            n1.shutdown(); n2.shutdown()
            return ok
        await asyncio.sleep(0.05)

    n1.shutdown(); n2.shutdown()
    return False


async def smoke_large() -> bool:
    n1, n2 = TensorNode(), TensorNode()
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    td = make_td(shape=(64, 64), randomish=True)
    n1.register_tensor("big", td)
    await n1.send_tensor(addr2, "big", td)

    for _ in range(400):
        if (r := await n2.receive_tensor()):
            ok = td_equal(td, r)
            n1.shutdown(); n2.shutdown()
            return ok
        await asyncio.sleep(0.05)

    n1.shutdown(); n2.shutdown()
    return False


async def smoke_address() -> bool:
    n = TensorNode()
    await n.start()
    addr = await n.get_node_addr()
    n.shutdown()
    return addr.startswith("node")


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

    for _ in range(50):
        if (r := await n2.receive_tensor()):
            same = torch.allclose(t, td_to_torch(r))
            n1.shutdown(); n2.shutdown()
            return same
        await asyncio.sleep(0.05)

    n1.shutdown(); n2.shutdown()
    return False


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

    for _ in range(400):
        if (r := await n2.receive_tensor()):
            same = torch.allclose(t, td_to_torch(r))
            n1.shutdown(); n2.shutdown()
            return same
        await asyncio.sleep(0.025)

    n1.shutdown(); n2.shutdown()
    return False


# ────────────── test harness ──────────────
async def main() -> None:
    tests = [
        ("Node addressing", smoke_address),
        ("Direct bytes tensor", smoke_direct),
        ("Large bytes tensor", smoke_large),
        ("Direct torch tensor", torch_direct),
        ("Large torch tensor", torch_large),
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
