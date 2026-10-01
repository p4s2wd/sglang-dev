"""Sustained-temperature cost of raising the power limit, on one card.

The fp16 GEMM sweep gives 1.31x at 250 W versus the current 150 W cap, but these
are modded 2080 Ti 22G cards: the memory mod shifts power onto the VRM, and
nvidia-smi's max_limit (280-330 W) describes what the regulator will accept, not
what the board dissipates safely forever. The real question is not "does it go
faster" (it does) but "does it still go faster after ten minutes", because a card
that reaches its thermal threshold starts thermal-throttling and gives back the
clock gain.

So: hold a saturating GEMM for 3 minutes at each candidate limit and report the
temperature and clock trajectory, not just the start. A level whose clock decays
over the run is a level that will not hold in production. GPU 0 only, restored
afterwards.
"""
import subprocess, sys, time
import torch

GPU = 0
LIM = int(float(subprocess.run(["nvidia-smi", "--query-gpu=power.limit",
                                "--format=csv,noheader,nounits", "-i", str(GPU)],
                               capture_output=True, text=True).stdout.strip()))


def q(fields):
    return subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
                           "--format=csv,noheader,nounits", "-i", str(GPU)],
                          capture_output=True, text=True).stdout.strip()


def setlim(w):
    subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(GPU), "-pl", str(w)],
                   capture_output=True, text=True)


print("current limit %d W" % LIM)
torch.cuda.set_device(GPU)
dev = torch.device("cuda:0")
a = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
b = torch.randn(8192, 8192, dtype=torch.float16, device=dev)
c = torch.empty(8192, 8192, dtype=torch.float16, device=dev)

for w in (LIM, 220, 250):
    setlim(w)
    time.sleep(2)
    print("\n=== limit %d W, 180 s sustained GEMM ===" % w)
    t_end = time.time() + 180
    i = 0
    while time.time() < t_end:
        for _ in range(150):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        i += 1
        if i % 10 == 0:
            clk, pw, tp = q("clocks.sm,power.draw,temperature.gpu").split(", ")[:3]
            print("  t+%3ds  clock %5s MHz  power %6s W  temp %3s C"
                  % (i * 12, clk, pw, tp), flush=True)
    clk, pw, tp = q("clocks.sm,power.draw,temperature.gpu").split(", ")[:3]
    print("  end state: clock %s MHz power %s W temp %s C" % (clk, pw, tp))
    time.sleep(20)
setlim(LIM)
print("\nrestored to %d W" % LIM)
