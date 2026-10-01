import os as _os, pathlib as _pb
def _find_repo():
    r = _os.environ.get("SGLANG_REPO")
    if r:
        return r
    d = _pb.Path(__file__).resolve().parent
    for _ in range(8):
        for cand in (d, d / "sglang"):
            if (cand / "python" / "sglang").is_dir():
                return str(cand)
        d = d.parent
    raise RuntimeError("set SGLANG_REPO to the sglang checkout")
_PY = _find_repo() + "/python"

import importlib.util, torch
spec = importlib.util.spec_from_file_location("m", _PY + "/sglang/kernels/ops/moe/mxfp4_w4a16_kernels.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev="cuda:0"; torch.manual_seed(0)
E,N,K=1,32,64
codes = torch.randint(0,16,(E,N,K),dtype=torch.int64)
packed=(codes[:,:,0::2] | (codes[:,:,1::2]<<4)).to(torch.uint8).view(torch.int8)
scale=torch.randint(118,138,(E,N,K//32),dtype=torch.uint8)  # random scales
a=torch.ones(1,K,device=dev).half()
w=packed.to(dev); s=scale.to(dev)
sorted_ids=torch.tensor([0],dtype=torch.int32,device=dev)
expert_ids=torch.tensor([0],dtype=torch.int32,device=dev)
out=torch.zeros(1,N,device=dev,dtype=torch.float16)
m.mxfp4_w4a16_gemm(a,w,s,sorted_ids,expert_ids,out,block_m=16,block_n=32,block_k=64)
torch.cuda.synchronize()
wref=m.dequant_mxfp4_reference(w,s)
ref=(a.float()@wref[0].t())[0]
print("random-scale single-iter: max abs err", (out.float()[0]-ref).abs().max().item())
print("out[:5]",out.float()[0,:5].tolist()); print("ref[:5]",ref[:5].tolist())
