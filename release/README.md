# Release bundles

Each subdirectory is one shipped release of the SM75 DeepSeek-V4-Flash path.
The full bundle (including the 16 MB wheel and the 16 MB tarball) lives on the
box at `/data/nvme/sglang/release-*` and is published as a GitHub Release
asset on `p4s2wd/sglang-sm75`; this directory tracks the text so the build
recipe, the launcher, the notes and the patches are versioned and diffable
without bloating git history with binaries.

To check a bundle against its published artifacts:

```sh
sha256sum -c SHA256SUMS
```

## sm75main2 — 2026-10-01

Base: upstream `main` @ `98fce73d5b`. Git tag `v0.2.0-sm75main2` @ `331faaeaf7`.
Wheel: `sglang-0.5.21.dev797+g331faaeaf7.sm75main2-py3-none-any.whl`.

**This is a correctness release, not a tuning release.** `sm75main1` and
everything older fault with an illegal memory access within the first few greedy
prompts on the stock configuration: `dsv4/topk.py`'s paged top-k transform took
`page_table_width` and never used it, so a stale token yielded a page id past the
end of the row and the sparse-attention gather used it as a KV address. Out of
range entries go 4 -> 0 with the bound in place.

The launcher also now defaults to `SGLANG_PP_LAYER_PARTITION=11,11,11,10`
(**+7.6% prefill**), which is the first time the F3 split shipped. The cost is
PP0's headroom, 0.93 GB -> 0.64 GB, the tightest rank in the fleet.

Ten commits since `sm75main1`. Seven are decode-side work done on the evening of
2026-09-30, *after* the sm75main1 wheel was cut, so they were never released;
three are the fix and its diagnostics.

## sm75main1 — 2026-09-30

Base: same `98fce73d5b`. Git tag `v0.2.0-sm75main1` @ `e0f76063c`. The original
rebase of the sub-90 work onto current main. **Superseded — it carries the IMA
bug described above.** Bundle on the box:
`/data/nvme/sglang/release-sm75-main/`.
