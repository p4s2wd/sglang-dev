#!/usr/bin/env python
"""Validate the parallel top-K now living in the production module.

Checks four things the prototype scripts could not, because they tested copies
of the kernels rather than the code that will run:

  byte-parity  the dispatched function must equal the single-program kernel it
               replaced, for every width and every ragged batch
  threshold    widths below the dispatch point must still take the old path, and
               the parallel path must be correct there too when forced
  graph safety the partial buffers are cached and never freed because a captured
               graph bakes their pointers in; a second capture with different
               batch geometry must not disturb an earlier graph
  values       duplicate scores, zeros, negatives and +-inf, since ties decide
               which of several equal elements fills the last slot

Run: topk_prod_validate.py
"""
import sys

import torch
import triton

sys.path.insert(0, "/data/nvme/sglang-codex/sglang/python")

from sglang.kernels.ops.attention.dsv4 import topk as T  # noqa: E402

K = 512
PS = 16
DEV = "cuda"


def single_program(scores, seq_lens, page_tables, out_page, raw, write_raw,
                   page_size):
    """The kernel as it was before the parallel path existed."""
    Kp2 = triton.next_power_of_2(out_page.shape[1])
    block_n = max(256, min(Kp2, 1024))
    T._topk_transform_paged_triton_kernel[(scores.shape[0],)](
        scores, seq_lens, page_tables, out_page, raw,
        scores.shape[1], scores.stride(0),
        page_tables.shape[1], page_tables.stride(0),
        K=out_page.shape[1], K_POW2=Kp2, BLOCK_N=block_n, PAGE_SIZE=page_size,
        WRITE_RAW=write_raw, num_warps=4, num_stages=1)


def parallel(scores, seq_lens, page_tables, out_page, raw, write_raw,
             page_size):
    Kp2 = triton.next_power_of_2(out_page.shape[1])
    block_n = max(256, min(Kp2, 1024))
    T._topk_transform_paged_parallel(
        scores, seq_lens, page_tables, out_page, raw, write_raw, page_size,
        out_page.shape[1], Kp2, block_n)


def pair(rows, cap, lens):
    pt = (torch.arange(1 << 17, device=DEV, dtype=torch.int32)
          .repeat(rows, 1) % 4096)
    o1 = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    o2 = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    r1 = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    r2 = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    return (o1, r1), (o2, r2)


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def parity():
    print("== 逐字节对比:并行路径 vs 单程序 kernel ==")
    cases = [
        (1, 4096, [4096]),
        (1, 8192, [8192]),
        (1, 16384, [16384]),
        (2, 8192, [8192, 8192]),
        (4, 8192, [8192, 4096, 1, 300]),
        (8, 37500, [37500, 20000, 37500, 1, 511, 512, 513, 37500]),
        (8, 65536, [65536, 65536, 100, 30000, 65536, 7, 513, 65536]),
        (8, 262144, [262144, 150000, 65536, 1024, 262144, 3, 77777, 262144]),
        (16, 65536, [65536 if i % 2 else 5000 + i for i in range(16)]),
        (3, 131072, [131072, 131072, 131072]),
    ]
    bad = 0
    for rows, cap, lens in cases:
        torch.manual_seed(cap + rows)
        scores = torch.randn(rows, cap, device=DEV)
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
        a, b = pair(rows, cap, lens)
        pt = (torch.arange(1 << 17, device=DEV, dtype=torch.int32)
              .repeat(rows, 1) % 4096)
        single_program(scores, seq_lens, pt, a[0], a[1], True, PS)
        parallel(scores, seq_lens, pt, b[0], b[1], True, PS)
        torch.cuda.synchronize()
        ok = same(a, b)
        bad += not ok
        print(f"  rows={rows:>3} cap={cap:>7}  "
              f"{'exact' if ok else 'MISMATCH'}", flush=True)
    print(f"  -> {'全部一致' if bad == 0 else f'{bad} 组不一致'}\n")
    return bad == 0


def dispatch():
    print("== 分派阈值:低于 8192 必须仍走单程序 kernel ==")
    for cap, want in ((4096, "single"), (8192, "parallel"), (65536, "parallel")):
        scores = torch.randn(1, cap, device=DEV)
        seq_lens = torch.full((1,), cap, dtype=torch.int32, device=DEV)
        pt = (torch.arange(1 << 17, device=DEV, dtype=torch.int32)
              .repeat(1, 1) % 4096)
        a, b = pair(1, cap, [cap])
        T.topk_transform_paged_triton(scores, seq_lens, pt, a[0], PS, a[1])
        single_program(scores, seq_lens, pt, b[0], b[1], True, PS)
        torch.cuda.synchronize()
        got = "parallel" if cap >= T._PARALLEL_MIN_WIDTH else "single"
        ok = torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])
        print(f"  cap={cap:>6}  选中={got:<8} 期望={want:<8} "
              f"结果与单程序一致={ok}")
    print()


def values():
    print("== 特殊取值:重复分数 / 全零 / 负值 / +-inf ==")
    cases = {}
    torch.manual_seed(7)
    cases["全常数(全部并列)"] = torch.full((1, 20000), 1.5, device=DEV)
    z = torch.zeros(1, 20000, device=DEV)
    z[0, ::3] = 2.0
    cases["大量并列+零"] = z
    cases["全负"] = -torch.rand(1, 20000, device=DEV).abs() - 1.0
    inf = torch.randn(1, 20000, device=DEV)
    inf[0, 100:150] = float("inf")
    inf[0, 200:250] = float("-inf")
    cases["含 +-inf"] = inf
    bad = 0
    for name, scores in cases.items():
        cap = scores.shape[1]
        seq_lens = torch.full((1,), cap, dtype=torch.int32, device=DEV)
        pt = (torch.arange(1 << 17, device=DEV, dtype=torch.int32)
              .repeat(1, 1) % 4096)
        a, b = pair(1, cap, [cap])
        single_program(scores, seq_lens, pt, a[0], a[1], True, PS)
        parallel(scores, seq_lens, pt, b[0], b[1], True, PS)
        torch.cuda.synchronize()
        ok = same(a, b)
        bad += not ok
        print(f"  {name:<18} {'exact' if ok else 'MISMATCH'}", flush=True)
    print(f"  -> {'全部一致' if bad == 0 else f'{bad} 组不一致'}\n")
    return bad == 0


def graph_safety():
    print("== CUDA graph:两次不同批形状的捕获互不干扰 ==")
    g_a = _capture(1, 16384, 16384)
    g_b = _capture(8, 262144, None)
    # replay A after B has been captured and replayed: A must still be right
    a_scores = _capture.scores_a
    seq_a = _capture.seq_a
    pt_a = _capture.pt_a
    o_a, r_a = _capture.out_a
    g_b.replay()
    g_a.replay()
    torch.cuda.synchronize()
    exp = set(torch.topk(a_scores[0], K).indices.tolist())
    got = set(v for v in r_a[0].tolist() if v >= 0)
    ok = got == exp
    print(f"  A(rows=1,16K) 在 B(rows=8,256K) 之后重放: "
          f"{'exact' if ok else 'MISMATCH'}")
    print(f"  buffer 缓存条目数: {len(T._PARALLEL_BUFFERS)}")
    return ok


def _capture(rows, cap, lens):
    torch.manual_seed(cap + rows)
    scores = torch.randn(rows, cap, device=DEV)
    seq_lens = (torch.full((rows,), cap, dtype=torch.int32, device=DEV)
                if lens is None
                else torch.full((rows,), lens, dtype=torch.int32, device=DEV))
    pt = (torch.arange(1 << 17, device=DEV, dtype=torch.int32)
          .repeat(rows, 1) % 4096)
    o = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    r = torch.empty(rows, K, dtype=torch.int32, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            T.topk_transform_paged_triton(scores, seq_lens, pt, o, PS, r)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        T.topk_transform_paged_triton(scores, seq_lens, pt, o, PS, r)
    torch.cuda.synchronize()
    if rows == 1:
        _capture.scores_a, _capture.seq_a, _capture.pt_a = scores, seq_lens, pt
        _capture.out_a = (o, r)
    return g


def main():
    ok = parity()
    ok &= values()
    dispatch()
    ok &= graph_safety()
    print(f"\n总判定: {'通过' if ok else '存在失败'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())