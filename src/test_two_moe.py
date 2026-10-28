import os, sys, llama_cpp
sys.stdout.reconfigure(encoding="utf-8")
print("llama_cpp (bundled llama.cpp):", llama_cpp.__version__)
M = os.path.join(os.path.dirname(__file__), "moe1", "granite-3.0-1b-a400m-instruct-Q4_K_M.gguf")

print("Loading MoE context A ...", flush=True)
a = llama_cpp.Llama(model_path=M, n_ctx=256, n_threads=4, verbose=False)
print("Loading MoE context B (a SECOND MoE model+context in the SAME process) ...", flush=True)
b = llama_cpp.Llama(model_path=M, n_ctx=256, n_threads=4, verbose=False)

for nm, mdl in (("A", a), ("B", b)):
    md = mdl.metadata
    arch = md.get("general.architecture")
    experts = {k: v for k, v in md.items() if "expert" in k.lower()}
    print(f"  context {nm}: arch={arch}  {experts}")

def gen(mdl, p):
    o = mdl.create_completion(p, max_tokens=20, temperature=0)
    return o["choices"][0]["text"].strip().replace("\n", " ")

print("--- interleaved decode on BOTH live MoE contexts (A,B,A,B) ---", flush=True)
print("A1:", repr(gen(a, "The capital of France is")))
print("B1:", repr(gen(b, "Two plus two equals")))
print("A2:", repr(gen(a, "Water is made of hydrogen and")))
print("B2:", repr(gen(b, "The sun rises in the")))
print("\nRESULT: two MoE contexts loaded + both producing tokens, interleaved, in ONE process = PASS")
