"""How much does the 150 W power cap cost, measured on one card?

Under prefill load the GPUs draw 130-185 W against power.limit=150 W (max_limit
280-330 W) and hold SM clocks at ~1350 MHz of a 2145 MHz maximum. If the cap is
what pins the clocks, raising it recovers clock speed and with it every
compute-bound kernel -- attention is 31% of prefill and runs on the mma path,
which is pure clock.

Measured on GPU 0 only, reversibly, with the server idle so nothing else is
affected: a fp16 GEMM loop sized to saturate the tensor cores, run at the current
limit and then at 250 W, interleaved so any drift shows up as spread rather than
direction. The GEMM is power-hungry by design -- a kernel that does not push the
card past 150 W cannot reveal the cap.
"""
import subprocess, sys, time
import torch

GPU = 0
LIM_DEFAULT = subprocess.run(["nvidia-smi", "--query-gpu=power.limit",
                              "--format=csv,noheader,nounits", "-i", str(GPU)],
                             capture_output=True, text=True).stdout.strip()


def set_limit(w):
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(GPU),
                    "-pl", str(w)], capture_output=True, text=True)


def read_state():
    o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                        "--format=csv,noheader,nounits", "-i", str(GPU)],
                       capture_output=True, text=True).stdout.strip()
    return o


torch.cuda.set_device(GPU)
dev = torch.device("cuda:0")
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
c = torch.empty(8192, 8192, dtype=torch.float16, device=dev)


def gemm_bench(iters=60):
    for _ in range(10):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tf = 2 * 8192 ** 3 / dt / 1e12
    return tf, read_state()


results = {}
LEVELS = (int(float(LIM_DEFAULT)), 180, 200, 220, 250)
for w in tuple(LEVELS) + tuple(LEVELS):
    set_limit(w)
    time.sleep(2)
    tf, st = gemm_bench()
    results.setdefault(w, []).append((tf, st))
    print("limit=%3d W -> %6.2f TFLOP/s fp16   (clock/power: %s)" % (w, tf, st), flush=True)
set_limit(int(float(LIM_DEFAULT)))
print("limit restored to %s W" % LIM_DEFAULT)

base = sorted(x[0] for x in results[LEVELS[0]])
b = base[len(base) // 2]
print("\npower limit -> fp16 GEMM (median of 2 sweep passes)")
for w in LEVELS:
    v = sorted(x[0] for x in results[w])
    m = v[len(v) // 2]
    print("  %3d W  %6.2f TFLOP/s  %.2fx vs current  (spread %.2f-%.2f)"
          % (w, m, m / b, v[0], v[-1]))
