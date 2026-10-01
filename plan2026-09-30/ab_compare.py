#!/usr/bin/env python
"""Interleaved A/B comparison across arms.

Both arms must have gone through `measure.sh <TAG>` so the round counts,
warmup, prompt set and load history are identical -- on this box the same
server measured 136.7 tok/s cold and 105.6 tok/s after a probe history, so a
mismatched history is worth more than any effect being tested.

Reports three things, because any one of them alone is misleading:
  1. the median and the within-arm spread (a cell whose spread is as large as
     the effect is not evidence),
  2. the ratio between arms,
  3. bs=1 as an internal control -- a config knob that changes decode
     scheduling should not move single-request latency, and if it does, the
     arms were probably not thermally comparable.

Usage: ab_compare.py A B [C ...]
"""
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")


def load_dec(tag):
    p = os.path.join(RES, f"dec_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # dec_probe.py writes {"port":..., "rounds":..., "summary": {"1": {...}}}
    # and the batch sizes arrive as strings, so int() the keys.
    return {int(k): v for k, v in d.get("summary", {}).items()}


def load_pf(tag):
    p = os.path.join(RES, f"pf_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def fmt(v, w=10):
    return f"{v:>{w}.1f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main():
    tags = sys.argv[1:] or ["A", "B"]
    decs = {t: load_dec(t) for t in tags}
    found = [t for t in tags if decs[t]]
    if not found:
        print("no res/dec_*.json found for any tag")
        return 1

    print("=" * 78)
    print("DECODE  (aggregate tok/s, median of rounds)")
    print("=" * 78)
    all_bs = sorted({b for t in found for b in decs[t]})
    print(f"{'bs':>4}" + "".join(f"{t + ' med':>11}{t + ' sprd':>9}" for t in found))
    for b in all_bs:
        row = f"{b:>4}"
        for t in found:
            cell = decs[t].get(b)
            if not cell:
                row += f"{'-':>11}{'-':>9}"
                continue
            med = cell["median"]
            rng = cell.get("all") or []
            spread = (max(rng) - min(rng)) / med * 100 if rng and med else 0
            row += f"{med:>11.1f}{spread:>8.1f}%"
        print(row)

    if len(found) >= 2:
        base, other = found[0], found[1]
        print()
        print(f"RATIO  {other}/{base}   (>1 = {other} faster)")
        print(f"{'bs':>4}{'ratio':>10}{'verdict':>16}")
        for b in all_bs:
            x, y = decs[base].get(b), decs[other].get(b)
            if not x or not y:
                continue
            r = y["median"] / x["median"]
            # a within-arm spread this wide means the cell cannot resolve the effect
            rng = y.get("all") or []
            sp = (max(rng) - min(rng)) / y["median"] * 100 if rng else 0
            note = "NOISY" if sp > 15 else ("faster" if r > 1.03 else "slower" if r < 0.97 else "same")
            print(f"{b:>4}{r:>10.3f}{note:>16}")
        b1x, b1y = decs[base].get(1), decs[other].get(1)
        if b1x and b1y:
            r = b1y["median"] / b1x["median"]
            print()
            print(f"CONTROL bs=1 ratio = {r:.3f}")
            print("  -> ~1.0 means the arms are thermally comparable and the")
            print("     multi-bs ratios below are attributable to the flag.")
            print("  -> far from 1.0 means the two arms were not measured under")
            print("     comparable conditions; interleave another round.")

    print()
    print("=" * 78)
    print("PREFILL  (aggregate tok/s, median of rounds)")
    print("=" * 78)
    pfs = {t: load_pf(t) for t in tags}
    pf_found = [t for t in tags if pfs[t]]
    if not pf_found:
        print("(no res/pf_*.json)")
        return 0
    # pf json is {"lens": [...], "ks": [...], "summary": {"L/K": {...}}}
    lens = pfs[pf_found[0]].get("lens", [])
    ks = pfs[pf_found[0]].get("ks", [])
    print(f"{'lens':>6}{'K':>4}" + "".join(f"{t:>11}" for t in pf_found))
    for L in lens:
        for k in ks:
            row = f"{L:>6}{k:>4}"
            for t in pf_found:
                cell = pfs[t].get("summary", {}).get(f"{L}/{k}")
                row += fmt(cell["median"] if cell else None)
            print(row)
        print()

    if len(pf_found) >= 2:
        b, o = pf_found[0], pf_found[1]
        print(f"PREFILL RATIO {o}/{b}")
        for L in lens:
            for k in ks:
                x = pfs[b].get("summary", {}).get(f"{L}/{k}")
                y = pfs[o].get("summary", {}).get(f"{L}/{k}")
                if x and y:
                    print(f"  L={L} K={k}: {y['median'] / x['median']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
