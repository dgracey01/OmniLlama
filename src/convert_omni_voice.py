#!/usr/bin/env python3
"""OmniLlama 5a: build the Qwen3-Omni voice (Code Predictor + Code2Wav) mmproj GGUF.

Direct-instantiation driver: constructs conversion.omnillama_voice.Qwen3OmniVoiceGenModel against
the real Omni checkpoint and writes an mmproj GGUF. Bypasses the conversion/__init__.py arch
dispatch (Omni's arch is already mapped to qwen3vl for the Thinker, so the stock CLI can't reach
the voice classes).

Run on Goliath (paths are Goliath-local; X:==I:, Z:==E:):
  I:\LLaMA-Factory\.venv\Scripts\python.exe -I I:\_bridge\convert_omni_voice.py
Needs: transformers (AutoConfig), torch, safetensors, numpy (LLaMA-Factory venv has them) + the
in-tree gguf-py/conversion (added to sys.path below, inserted first so my constants edit wins).
"""
import os, sys

LLAMACPP = r"I:\llamacpp"
MODEL    = r"E:\Qwen3-Omni-30B-A3B-Instruct"
OUT_DIR  = r"I:\_bridge\omni_voice_out"

sys.path.insert(0, os.path.join(LLAMACPP, "gguf-py"))  # the in-tree `gguf` package (has A_GEN_WAV_CODE_EMBD)
sys.path.insert(0, LLAMACPP)                           # the `conversion` package
sys.stdout.reconfigure(encoding="utf-8")

os.makedirs(OUT_DIR, exist_ok=True)

from pathlib import Path
import gguf
print("gguf from:", gguf.__file__, flush=True)
assert hasattr(gguf.MODEL_TENSOR, "A_GEN_WAV_CODE_EMBD"), "constants.py edit not picked up"

from conversion.omnillama_voice import Qwen3OmniVoiceGenModel

print(f"converting voice mmproj: {MODEL} -> {OUT_DIR}", flush=True)
model = Qwen3OmniVoiceGenModel(
    Path(MODEL),
    gguf.LlamaFileType.MOSTLY_F16,
    Path(OUT_DIR),
    eager=False,
)
model.write()
print("DONE. mmproj GGUF written under:", OUT_DIR, flush=True)
for f in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, f)
    print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
