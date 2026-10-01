"""Does the sub-90 model_hook default survive argument resolution?"""
import os, sys
from sglang.srt.server_args import ServerArgs

sa = ServerArgs(
    model_path=os.environ["DSV4_CKPT"],
    tp_size=2, pp_size=4,
    mem_fraction_static=0.97,
    kv_cache_dtype="fp8_e4m3",
    context_length=1024,
    chunked_prefill_size=512,
    max_running_requests=1,
    cuda_graph_backend_decode="disabled",
    disable_prefill_cuda_graph=True,
)
sa.resolve_once()
print("dsa_topk_backend   :", sa.dsa_topk_backend)
print("moe_runner_backend :", sa.moe_runner_backend)
from sglang.srt.environ import envs
print("DSA_FUSE_TOPK      :", envs.SGLANG_DSA_FUSE_TOPK.get())
from sglang.srt.runtime_context import get_exec, publish
publish(sa, role="engine")
print("exec.kernel.dsa_topk_backend:", get_exec().kernel.dsa_topk_backend)
