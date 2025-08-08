#!/usr/bin/env python3
"""
bench_receiver.py – RTT benchmark reflector
Launch first on the *remote* box.

It prints its NodeTicket, then waits for “PING” tensors.
For every PING it immediately ships the payload back to the
originating address embedded in the tensor bytes – giving your sender
a full round-trip measurement.
"""

import asyncio, sys, time
import tensor_iroh as tp

# ───── helpers ──────────────────────────────────────────────────────────
def parse_ping(td: tp.PyTensorData) -> tuple[str, bytes]:
    """
    Tensor layout:  b'ADDR:<sender_ticket>\n' + payload
    Returns (sender_ticket, payload_bytes)
    """
    raw = td.as_bytes()
    if not raw.startswith(b"ADDR:"):
        raise ValueError("malformed ping (missing ADDR header)")
    addr_end = raw.find(b"\n")
    if addr_end == -1:
        raise ValueError("malformed ping (no newline)")
    sender_addr = raw[5:addr_end].decode()
    payload = raw[addr_end + 1 :]
    return sender_addr, payload


def build_pong(payload: bytes) -> tp.PyTensorData:
    return tp.PyTensorData(payload, [len(payload)], "float16", False)


# ───── main loop ────────────────────────────────────────────────────────
async def main() -> None:
    node = tp.PyTensorNode()
    await node.start()
    my_addr = await node.get_node_addr()
    print(f"🏁 Reflector ready – give this to your sender:\n{my_addr}\n", flush=True)
    await asyncio.sleep(0.2)  # let discovery settle without blocking the event loop

    while True:
        name, td = await node.wait_for_tensor()
        sender_addr, payload = parse_ping(td)
        pong = build_pong(payload)               # echo identical bytes
        # name can be constant – Response message carries data directly
        await node.send_tensor(sender_addr, "PONG", pong)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
