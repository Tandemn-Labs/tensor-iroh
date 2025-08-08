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
    header: bytes,
) -> float:
    """Send one ping, wait for its pong, return RTT in milliseconds."""
    blob = header + payload
    td = tp.PyTensorData(blob, [len(blob)], "uint8", False)

    start_ns = time.perf_counter_ns()
    await node.send_tensor(receiver, f"PING{idx}", td)

    # wait for pong without polling
    _name, _pong_td = await node.wait_for_tensor()

    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1e6   # → ms


async def main() -> None:
    ap = argparse.ArgumentParser(description="Tensor-RTT benchmark")
    ap.add_argument("receiver", help="NodeTicket of remote reflector")
    ap.add_argument("--size", type=int, default=65500,
                    help="payload bytes (default 64 KiB)")
    ap.add_argument("--count", type=int, default=1000,
                    help="number of pings (default 1000)")
    args = ap.parse_args()

    node = tp.PyTensorNode()
    await node.start()
    print("⏳ warming connection…")
    await asyncio.sleep(0.2)                      # let discovery finish

    sender_addr = await node.get_node_addr()
    header = HEADER_TEMPLATE % sender_addr.encode()

    payload = secrets.token_bytes(args.size)
    rtts = []

    for i in range(args.count):
        rtt = await ping_once(node, args.receiver, payload, i, header)
        rtts.append(rtt)
        print(f"Ping {i:03}: {rtt:.2f} ms")
    # Remove outliers using IQR method
    q1 = statistics.quantiles(rtts, n=4)[0]  # 25th percentile
    q3 = statistics.quantiles(rtts, n=4)[2]  # 75th percentile
    iqr = q3 - q1
    lower_bound = q1 - 1.8 * iqr
    upper_bound = q3 + 1.8 * iqr
    
    filtered_rtts = [rtt for rtt in rtts if lower_bound <= rtt <= upper_bound]
    outliers_removed = len(rtts) - len(filtered_rtts)

    print("\n━━ results ━━")
    print(f" samples        : {args.count} ({outliers_removed} outliers removed)")
    print(f" min | med | max: "
          f"{min(filtered_rtts):.2f} | {statistics.median(filtered_rtts):.2f} | {max(filtered_rtts):.2f} ms")
    print(f" mean ± stdev   : "
          f"{statistics.mean(filtered_rtts):.2f} ± {statistics.stdev(filtered_rtts):.2f} ms")

    # Create a simple ASCII histogram
    print("\n━━ RTT distribution ━━")
    bins = 20
    min_rtt, max_rtt = min(filtered_rtts), max(filtered_rtts)
    bin_width = (max_rtt - min_rtt) / bins
    
    histogram = [0] * bins
    for rtt in filtered_rtts:
        bin_idx = min(int((rtt - min_rtt) / bin_width), bins - 1)
        histogram[bin_idx] += 1
    
    max_count = max(histogram)
    scale = 50 / max_count if max_count > 0 else 1
    
    for i, count in enumerate(histogram):
        bin_start = min_rtt + i * bin_width
        bin_end = min_rtt + (i + 1) * bin_width
        bar_length = int(count * scale)
        bar = "█" * bar_length
        print(f"{bin_start:6.1f}-{bin_end:6.1f}ms │{bar:<50}│ {count:3d}")

if __name__ == "__main__":
    asyncio.run(main())
