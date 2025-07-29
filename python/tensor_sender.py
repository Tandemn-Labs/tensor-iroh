#!/usr/bin/env python3
"""
bench_sender.py – RTT benchmark initiator
Run on your *local* box, passing the receiver’s NodeTicket.

Example:
    python bench_sender.py node1abc… --size 65536 --count 100
"""

import asyncio, argparse, secrets, statistics, time
import tensor_iroh as tp

HEADER_TEMPLATE = b"ADDR:%b\n"      # %b → sender ticket as bytes


async def ping_once(
    node: tp.PyTensorNode,
    receiver: str,
    payload: bytes,
    idx: int,
) -> float:
    """Send one ping, wait for its pong, return RTT in milliseconds."""
    header = HEADER_TEMPLATE % (await node.get_node_addr()).encode()
    blob = header + payload
    td = tp.PyTensorData(blob, [len(blob)], "uint8", False)

    start_ns = time.perf_counter_ns()
    await node.send_tensor(receiver, f"PING{idx}", td)

    # wait for pong
    while True:
        pong = await node.receive_tensor()
        if pong is not None:
            break
        await asyncio.sleep(0.0001)

    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1e6   # → ms


async def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor-RTT benchmark")
    ap.add_argument("receiver", help="NodeTicket of remote reflector")
    ap.add_argument("--size", type=int, default=4096,
                    help="payload bytes (default 4 KiB)")
    ap.add_argument("--count", type=int, default=50,
                    help="number of pings (default 50)")
    args = ap.parse_args()

    node = tp.PyTensorNode()
    await node.start()
    print("⏳ warming connection…")
    await asyncio.sleep(0.2)                      # let discovery finish

    payload = secrets.token_bytes(args.size)
    rtts = []

    for i in range(args.count):
        rtt = await ping_once(node, args.receiver, payload, i)
        rtts.append(rtt)
        print(f"Ping {i:03}: {rtt:.2f} ms")

    print("\n━━ results ━━")
    print(f" samples        : {args.count}")
    print(f" min | med | max: "
          f"{min(rtts):.2f} | {statistics.median(rtts):.2f} | {max(rtts):.2f} ms")
    print(f" mean ± stdev   : "
          f"{statistics.mean(rtts):.2f} ± {statistics.stdev(rtts):.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
