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
E,N,K,M=8,128,256,37
codes = torch.randint(0,16,(E,N,K),dtype=torch.int64)
packed=(codes[:,:,0::2] | (codes[:,:,1::2]<<4)).to(torch.uint8).view(torch.int8)
scale=torch.randint(118,128,(E,N,K//32),dtype=torch.uint8)
a=(torch.randn(M,K,device=dev)*0.1).half()
w=packed.to(dev); s=scale.to(dev)
token_experts=torch.randint(0,E,(M,),device=dev)
block_m=16
# per-expert padded layout (moe_align semantics): each expert's tokens padded to block_m multiple
counts=torch.bincount(token_experts,minlength=E)
blocks=(counts+block_m-1)//block_m
expert_ids=torch.repeat_interleave(torch.arange(E,dtype=torch.int32,device=dev),blocks.to(torch.int32))
parts=[]
for e in range(E):
    idx=torch.nonzero(token_experts==e).flatten().to(torch.int32)
    pad=int(blocks[e])*block_m-idx.numel()
    parts.append(torch.cat([idx,torch.full((pad,),M,dtype=torch.int32,device=dev)]))
sorted_ids=torch.cat(parts)
out=torch.zeros(M,N,device=dev,dtype=torch.float16)
m.mxfp4_w4a16_gemm(a,w,s,sorted_ids,expert_ids,out)
torch.cuda.synchronize()
wref=m.dequant_mxfp4_reference(w,s)
ref=torch.stack([a[mm].float()@wref[int(token_experts[mm])].t() for mm in range(M)])
err=(out.float()-ref).abs().max(dim=1).values
bad=(err>0.5).nonzero().flatten().tolist()
print("bad rows:",bad[:20],"of",M,"max err",err.max().item())
