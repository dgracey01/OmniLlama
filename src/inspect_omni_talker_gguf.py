#!/usr/bin/env python3
"""OmniLlama 5b-1 structural check of the talker backbone GGUF (reads the file directly; no llama.cpp)."""
import os, sys, glob
sys.path.insert(0, r"I:\llamacpp\gguf-py")
sys.stdout.reconfigure(encoding="utf-8")
from gguf import GGUFReader

OUT = r"I:\_bridge\omni_talker_out"
cand = sorted(glob.glob(os.path.join(OUT, "*.gguf")))
assert cand, f"no gguf in {OUT}"
p = cand[-1]
print("file:", p, f"({os.path.getsize(p)/1e6:.1f} MB)")
r = GGUFReader(p)

print("\n=== KV (selected) ===")
for f in r.fields.values():
    nm = f.name
    if any(s in nm for s in ("architecture", "block_count", "embedding_length", "head_count",
                             "feed_forward", "expert", "vocab", "context_length", "rope", "key_length",
                             "value_length", "layer_norm", "eos", "tokenizer.ggml.model")):
        try: v = f.contents()
        except Exception: v = "<?>"
        print(f"  {nm} = {v}")

T = {t.name: tuple(int(x) for x in t.shape) for t in r.tensors}
print(f"\n=== {len(T)} tensors ===")
def show(name):
    print(f"  {name:40s} {T.get(name, 'MISSING')}")
for n in ("token_embd.weight", "output_norm.weight", "output.weight"):
    show(n)
print("  -- blk.0 --")
for suf in ("attn_norm", "attn_q", "attn_q_norm", "attn_k", "attn_k_norm", "attn_v", "attn_output",
            "ffn_norm", "ffn_gate_inp", "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps",
            "ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp"):
    show(f"blk.0.{suf}.weight")

# counts + asserts
nblk = len({n.split(".")[1] for n in T if n.startswith("blk.")})
print(f"\n=== blk layers present: {nblk} ===")
checks = {
    "token_embd 3072 vocab": T.get("token_embd.weight", (0, 0))[1] == 3072,
    "output 3072 vocab":     T.get("output.weight", (0, 0))[1] == 3072,
    "20 layers":             nblk == 20,
    "experts stacked x128":  T.get("blk.0.ffn_gate_exps.weight", (0, 0, 0))[-1] == 128,
    "shared expert present": "blk.0.ffn_gate_shexp.weight" in T,
    "q_norm present":        "blk.0.attn_q_norm.weight" in T,
}
print("\n=== CHECKS ===")
ok = True
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = ok and v
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
