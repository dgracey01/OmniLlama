#!/usr/bin/env python3
r"""OmniLlama 5b-5 Tier-2 driver: build the text front-end GGUF (arch qwen3vlmoe, vocab + token_embd +
text_projection + codec_special_embd). Direct-instantiation, bypasses conversion/__init__.py dispatch.
Run on Goliath:  I:\LLaMA-Factory\.venv\Scripts\python.exe -I I:\_bridge\convert_omni_textfe.py
"""
import os, sys

LLAMACPP = r"I:\llamacpp"
MODEL    = r"E:\Qwen3-Omni-30B-A3B-Instruct"
OUT_DIR  = r"I:\_bridge\omni_fe_out"

sys.path.insert(0, os.path.join(LLAMACPP, "gguf-py"))
sys.path.insert(0, LLAMACPP)
sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(OUT_DIR, exist_ok=True)

from pathlib import Path
import gguf
print("gguf from:", gguf.__file__, flush=True)

from conversion.omnillama_textfe import Qwen3OmniTextFrontEnd

print(f"converting text front-end: {MODEL} -> {OUT_DIR}", flush=True)
model = Qwen3OmniTextFrontEnd(Path(MODEL), gguf.LlamaFileType.MOSTLY_F16, Path(OUT_DIR), eager=False)
model.write()
print("DONE. front-end GGUF written under:", OUT_DIR, flush=True)
for f in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, f)
    print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
