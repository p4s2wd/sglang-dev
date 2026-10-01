"""Sustained 250 W on GPU 0: does the clock gain hold after minutes of load?

The sweep gives 1.31x at 250 W versus the 150 W cap, but these are modded 2080 Ti
22G cards and nvidia-smi's max_limit describes what the regulator accepts, not what
the board sheds safely forever. A level that reaches the thermal threshold starts
thermal-throttling and gives the clock back, so the number that matters is the
clock at the END of a long load, not the start.

300 s of saturating fp16 GEMM at 250 W, sampling every ~24 s. GPU 0 only; the limit
is restored even if the process is killed, via atexit and a signal handler.
"""
import atexit, signal, subprocess, time
import torch

GPU = 0
LIM = 150


def setlim(w):
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(GPU), "-pl", str(w)],
                   capture_output=True, text=True)


def restore():
    setlim(LIM)
    print("restored to %d W" % LIM, flush=True)


atexit.register(restore)
signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit()))

torch.cuda.set_device(GPU)
dev = torch.device("cuda:0")
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
c = torch.empty(8192, 8192, dtype=torch.float16, device=dev)

setlim(250)
print("=== GPU0 at 250 W, 300 s sustained GEMM ===", flush=True)
t_end = time.time() + 300
i = 0
while time.time() < t_end:
    t0 = time.perf_counter()
    for _ in range(150):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 150
    tf = 2 * 8192 ** 3 / dt / 1e12
    o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits", "-i", str(GPU)],
                       capture_output=True, text=True).stdout.strip().split(", ")
    print("  t+%3ds clock %5s MHz power %6s W temp %3s C  %6.2f TFLOP/s"
          % (i * 24, o[0], o[1], o[2], tf), flush=True)
    i += 1
