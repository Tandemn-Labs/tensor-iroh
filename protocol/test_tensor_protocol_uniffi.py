#!/usr/bin/env python3
"""
Tensor-Protocol integration test-bench
======================================
No PyO3 bindings are used here (yet) – just UniFFI.
"""

import asyncio, time, struct, logging, random
from typing import Optional, Tuple

# ────────────────────────── UniFFI bindings ──────────────────────────
try:
    from tensor_protocol_py import TensorNode, TensorData, TensorMetadata, create_node
    from tensor_protocol_py.tensor_protocol import uniffi_set_event_loop
except ImportError as e:        # pragma: no cover
    raise SystemExit(
        "❌  UniFFI bindings not found – build the Rust crate first "
        "(cargo build --release && ./build_bindings.sh)"
    ) from e

# ───────────────────────────── Torch (optional) ──────────────────────
try:
    import torch
    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False
    torch = None            # type: ignore

# ───────────────────────────── helpers ───────────────────────────────
def setup_event_loop() -> None:
    """Tell UniFFI which asyncio loop we’re using."""
    try:
        uniffi_set_event_loop(asyncio.get_running_loop())
    except RuntimeError:
        # called outside an event loop – nothing to do, asyncio.run() will
        # install the loop before any UniFFI call happens.
        pass

# ────────────────────────── torch helpers ────────────────────────────
def torch_to_td(t: "torch.Tensor") -> TensorData:
    if t.dtype != torch.float32:
        raise ValueError("only float32 tensors supported")
    if not t.is_contiguous():
        t = t.contiguous()
    return TensorData(
        metadata=TensorMetadata(
            shape=list(t.shape),
            dtype="float32",
            requires_grad=bool(t.requires_grad)),
        data=t.cpu().numpy().tobytes()
    )


def td_to_torch(td: TensorData) -> "torch.Tensor":
    import numpy as np
    arr = np.frombuffer(td.data, dtype=np.float32).reshape(td.metadata.shape)
    return torch.from_numpy(arr).clone()     # clone → owns its buffer


def td_equal(a: TensorData, b: TensorData) -> bool:
    return (a.metadata.shape == b.metadata.shape
            and a.metadata.dtype == b.metadata.dtype
            and a.data == b.data)


# ───────────────────────── tests ─────────────────────────────────────
async def smoke_direct() -> bool:
    node1, node2 = create_node(None), create_node(None)
    await node1.start(); await node2.start()

    addr2 = await node2.get_node_addr()
    td = make_tensor_data()

    node1.register_tensor("t", td)
    await node1.send_tensor_direct(addr2, "t", td)

    for _ in range(50):
        if (r := await node2.receive_tensor()):
            ok = td_equal(td, r)
            node1.shutdown(); node2.shutdown()
            return ok
        await asyncio.sleep(.05)

    node1.shutdown(); node2.shutdown()
    return False


async def smoke_large() -> bool:
    node1, node2 = create_node(None), create_node(None)
    await node1.start(); await node2.start()
    addr2 = await node2.get_node_addr()

    td = make_tensor_data(shape=(64, 64), randomish=True)
    node1.register_tensor("big", td)
    await node1.send_tensor_direct(addr2, "big", td)

    for _ in range(400):
        if (r := await node2.receive_tensor()):
            ok = td_equal(td, r)
            node1.shutdown(); node2.shutdown()
            return ok
        await asyncio.sleep(.05)

    node1.shutdown(); node2.shutdown()
    return False


async def smoke_address() -> bool:
    n = create_node(None)
    await n.start()
    addr = await n.get_node_addr()
    n.shutdown()
    return addr.startswith("node")


# ───────────────────────── torch tests ───────────────────────────────
async def torch_direct() -> bool:
    if not _HAS_TORCH:      # pragma: no cover
        print("ℹ️  torch not installed – skipping torch_direct")
        return True

    n1, n2 = create_node(None), create_node(None)
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    td = torch_to_td(t)
    n1.register_tensor("torch", td)
    await n1.send_tensor_direct(addr2, "torch", td)

    for _ in range(50):
        if (r := await n2.receive_tensor()):
            same = torch.allclose(t, td_to_torch(r))
            n1.shutdown(); n2.shutdown()
            return same
        await asyncio.sleep(.05)

    n1.shutdown(); n2.shutdown()
    return False


async def torch_large() -> bool:
    if not _HAS_TORCH:      # pragma: no cover
        print("ℹ️  torch not installed – skipping torch_large")
        return True

    n1, n2 = create_node(None), create_node(None)
    await n1.start(); await n2.start()
    addr2 = await n2.get_node_addr()

    t = torch.randn(128, 128, dtype=torch.float32)
    td = torch_to_td(t)
    n1.register_tensor("torch_big", td)
    await n1.send_tensor_direct(addr2, "torch_big", td)

    for _ in range(400):
        if (r := await n2.receive_tensor()):
            same = torch.allclose(t, td_to_torch(r))
            n1.shutdown(); n2.shutdown()
            return same
        await asyncio.sleep(.025)

    n1.shutdown(); n2.shutdown()
    return False


# ───────────────────────── harness ───────────────────────────────────
async def main() -> None:
    setup_event_loop()

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
        passed += ok

    print(f"\n=== {passed}/{len(tests)} tests passed ===")
    if passed != len(tests):
        raise SystemExit(1)



import struct, random, math      # add math import

# ─────────────────────── fixed helpers ──────────────────────────
def tensor_bytes(shape):
    """deterministic float32 ramp – torch-free version."""
    n = math.prod(shape)             # pure-Python product
    ba = bytearray()
    for i in range(n):
        ba.extend(struct.pack('<f', float(i)))
    return bytes(ba)


def make_tensor_data(shape=(3, 4), *, dtype="float32", randomish=False):
    if dtype != "float32":
        raise ValueError("Only float32 supported")

    if randomish:
        ba = bytearray()
        for i in range(math.prod(shape)):
            # deterministic-but-varied value
            val = float(i) + random.random()
            ba.extend(struct.pack('<f', val))
        raw = bytes(ba)
    else:
        raw = tensor_bytes(shape)

    meta = TensorMetadata(shape=list(shape), dtype=dtype, requires_grad=False)
    return TensorData(metadata=meta, data=raw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    asyncio.run(main())

