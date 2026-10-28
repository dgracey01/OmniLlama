#!/usr/bin/env python3
"""OmniLlama Phase 2a (MEASURE): dump the Omni voice-module tensor names + shapes + dtypes.

Run on Goliath (model is local via E:; needs `safetensors`):
    python inspect_code2wav.py --model "Z:/Qwen3-Omni-30B-A3B-Instruct" > code2wav_shapes.txt

Purpose: the one remaining reconciliation before writing the code2wav converter is to diff
Omni's `code2wav.*` layout against the A_GEN_WAV_* graph inputs expected by
`X:/llamacpp/tools/mtmd/models/qwen3tts-gen.cpp::code2wav::decode`. TTS reads a SEPARATE
speech_tokenizer/model.safetensors with `decoder.*` names (incl. RVQ EMA codebooks
`decoder.quantizer.rvq_first/rvq_rest`). Omni ships `code2wav.*` in the MAIN safetensors and
(per the index) shows `code2wav.code_embedding` with NO `quantizer.rvq_*` prefix — so the
quantizer/codec-decode handling may differ. This script surfaces exactly what Omni has.
"""
import argparse, json, os, struct, sys
from collections import defaultdict

VOICE_PREFIXES = ("code2wav.", "talker.")  # talker.* includes talker.code_predictor.*

def read_safetensors_header(path):
    """Return {tensor_name: {dtype, shape}} from a .safetensors header (no tensor data loaded)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n).decode("utf-8"))
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        out[k] = {"dtype": v.get("dtype"), "shape": v.get("shape")}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (safetensors sharded)")
    args = ap.parse_args()
    md = args.model
    idx_path = os.path.join(md, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path, encoding="utf-8"))["weight_map"]

    # which shards hold voice tensors
    shards = sorted({weight_map[k] for k in weight_map
                     if any(k.startswith(p) for p in VOICE_PREFIXES)})
    info = {}
    for sh in shards:
        info.update(read_safetensors_header(os.path.join(md, sh)))

    groups = defaultdict(list)
    for k in sorted(weight_map):
        if not any(k.startswith(p) for p in VOICE_PREFIXES):
            continue
        meta = info.get(k, {})
        # bucket by the module (code2wav / talker.code_predictor / talker)
        if k.startswith("code2wav."):
            g = "code2wav"
        elif k.startswith("talker.code_predictor."):
            g = "talker.code_predictor"
        else:
            g = "talker(backbone)"
        groups[g].append((k, meta.get("dtype"), meta.get("shape")))

    for g in ("code2wav", "talker.code_predictor", "talker(backbone)"):
        rows = groups.get(g, [])
        print(f"\n################ {g}  ({len(rows)} tensors) ################")
        for k, dt, shp in rows:
            print(f"  {k:70s} {str(dt):8s} {shp}")

    # reconciliation flags
    c2w = [k for k, *_ in groups.get("code2wav", [])]
    print("\n================ RECONCILIATION FLAGS (for the code2wav converter) ================")
    print("  has code_embedding :", any("code_embedding" in k for k in c2w))
    print("  has quantizer/rvq  :", any(("quantizer" in k or "rvq" in k) for k in c2w),
          "  <- TTS uses decoder.quantizer.rvq_first/rvq_rest; if False, Omni decodes codes differently")
    print("  has pre_transformer:", any("pre_transformer" in k for k in c2w))
    print("  has upsample       :", any("upsample" in k for k in c2w))
    print("  has decoder blocks :", any(k.startswith("code2wav.decoder.") for k in c2w))
    print("\nNext: map each code2wav.* name -> the A_GEN_WAV_* tensor it feeds in "
          "qwen3tts-gen.cpp::code2wav (compare to qwen3tts.py::_generate_code2wav_tensors).")

if __name__ == "__main__":
    main()
