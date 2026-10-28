#!/usr/bin/env python3
"""OmniLlama 5b-1 driver: build the Qwen3-Omni Talker backbone text GGUF (arch qwen3omnitalker).

Direct-instantiation (bypasses conversion/__init__.py dispatch, same as the 5a voice driver).
Run on Goliath:  I:\LLaMA-Factory\.venv\Scripts\python.exe -I I:\_bridge\convert_omni_talker.py
"""
import os, sys

LLAMACPP = r"I:\llamacpp"
MODEL    = r"E:\Qwen3-Omni-30B-A3B-Instruct"
OUT_DIR  = r"I:\_bridge\omni_talker_out"

sys.path.insert(0, os.path.join(LLAMACPP, "gguf-py"))  # in-tree gguf (has QWEN3OMNITALKER)
sys.path.insert(0, LLAMACPP)                           # conversion package
sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(OUT_DIR, exist_ok=True)

from pathlib import Path
import gguf
print("gguf from:", gguf.__file__, flush=True)
assert hasattr(gguf.MODEL_ARCH, "QWEN3OMNITALKER"), "constants.py edit not picked up"

from conversion.omnillama_talker import Qwen3OmniTalkerModel

print(f"converting talker backbone: {MODEL} -> {OUT_DIR}", flush=True)
model = Qwen3OmniTalkerModel(Path(MODEL), gguf.LlamaFileType.MOSTLY_F16, Path(OUT_DIR), eager=False)
model.write()
print("DONE. talker GGUF written under:", OUT_DIR, flush=True)
for f in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, f)
    print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
