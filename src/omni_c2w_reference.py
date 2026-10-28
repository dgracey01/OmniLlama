"""OmniLlama 5a — GOLD reference generator for Code2Wav (runs on Goliath 3090, LLaMA-Factory venv).

Loads ONLY the code2wav.* weights from the Qwen3-Omni checkpoint into the transformers
Qwen3OmniMoeCode2Wav module, runs forward() on a fixed seeded 16xT codec-token sequence, and dumps:
  - codes.npy        : the input codes [1,16,T]  (int64)
  - ref_wav.npy      : final waveform [1,1,N] float32  (the yardstick for the llama.cpp side)
  - ref.wav          : 24 kHz PCM16 so Zero can LISTEN (should be coherent audio texture, not silence)
  - stages.npz       : per-stage intermediates for stage-by-stage C++ debugging:
                         after_embed_mean [1,T,1024], after_pretfm [1,T,1024],
                         after_upsample [1,1024,N'], final_wav [1,1,N]
  - meta.json        : shapes, dtypes, config echo, checksums

Run:  I:\LLaMA-Factory\.venv\Scripts\python.exe -I I:\_bridge\omni_c2w_reference.py
"""
import os, sys, json, wave, struct, hashlib
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch

MODEL = r"E:\Qwen3-Omni-30B-A3B-Instruct"   # Goliath-local (TheBeast's Z: == Goliath's E:)
OUT   = r"I:\_bridge\c2w_ref"
os.makedirs(OUT, exist_ok=True)
T_FRAMES = 64            # 64 code frames -> 64*1920 = 122880 samples = ~5.1 s at 24 kHz
SEED     = 1234

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav
from safetensors import safe_open

print("loading config ...", flush=True)
full_cfg = json.load(open(os.path.join(MODEL, "config.json"), "r", encoding="utf-8"))
c2w_dict = dict(full_cfg["code2wav_config"])
c2w_dict.pop("model_type", None)   # empty string in the json; let the class default it
c2w_cfg = Qwen3OmniMoeCode2WavConfig(**c2w_dict)
print("  code2wav hidden=%d layers=%d heads=%d kv=%d win=%d nq=%d cb=%d up_rates=%s up_ratios=%s layer_types=%s" % (
    c2w_cfg.hidden_size, c2w_cfg.num_hidden_layers, c2w_cfg.num_attention_heads,
    getattr(c2w_cfg, "num_key_value_heads", -1), getattr(c2w_cfg, "sliding_window", -1),
    c2w_cfg.num_quantizers, c2w_cfg.codebook_size, c2w_cfg.upsample_rates, c2w_cfg.upsampling_ratios,
    getattr(c2w_cfg, "layer_types", None)), flush=True)

print("instantiating Qwen3OmniMoeCode2Wav ...", flush=True)
model = Qwen3OmniMoeCode2Wav(c2w_cfg)

# --- load ONLY code2wav.* weights from the sharded safetensors ---
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json"), "r", encoding="utf-8"))["weight_map"]
want = {k: v for k, v in idx.items() if k.startswith("code2wav.")}
print(f"  code2wav tensors to load: {len(want)}", flush=True)
sd = {}
by_shard = {}
for k, shard in want.items():
    by_shard.setdefault(shard, []).append(k)
for shard, keys in by_shard.items():
    with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
        for k in keys:
            sd[k[len("code2wav."):]] = f.get_tensor(k).float()   # fp32 reference
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if "code_offset" not in m]   # code_offset is a non-persistent buffer
print(f"  load_state_dict: missing(non-buffer)={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing:    print("    MISSING:", missing[:20], flush=True)
if unexpected: print("    UNEXPECTED:", unexpected[:20], flush=True)

model = model.float().eval().cuda()

# --- fixed seeded codes [1, 16, T], each in [0, codebook_size-1] ---
g = torch.Generator().manual_seed(SEED)
codes = torch.randint(0, c2w_cfg.codebook_size, (1, c2w_cfg.num_quantizers, T_FRAMES), generator=g).cuda()

# --- forward with per-stage taps (mirror of the real forward) ---
stages = {}
with torch.no_grad():
    code_offset = model.code_offset.to(codes.device)
    hidden = model.code_embedding(codes + code_offset).mean(1)              # [1,T,1024]
    stages["after_embed_mean"] = hidden.detach().float().cpu().numpy()
    hidden = model.pre_transformer(inputs_embeds=hidden).last_hidden_state  # [1,T,1024]
    stages["after_pretfm"] = hidden.detach().float().cpu().numpy()
    hidden = hidden.permute(0, 2, 1)                                        # [1,1024,T]
    for blocks in model.upsample:
        for block in blocks:
            hidden = block(hidden)
    stages["after_upsample"] = hidden.detach().float().cpu().numpy()
    wav = hidden
    for block in model.decoder:
        wav = block(wav)
    wav = wav.clamp(-1, 1)                                                  # [1,1,N]
    stages["final_wav"] = wav.detach().float().cpu().numpy()

    # also the library's own one-call path, as an internal cross-check
    wav_ref = model(codes).clamp(-1, 1).detach().float().cpu().numpy()

wav_np = stages["final_wav"]
agree = float(np.max(np.abs(wav_np - wav_ref)))
print(f"  manual-vs-model() max abs diff: {agree:.3e}  (should be ~0)", flush=True)

# --- save artifacts ---
np.save(os.path.join(OUT, "codes.npy"), codes.detach().cpu().numpy().astype(np.int64))
np.save(os.path.join(OUT, "ref_wav.npy"), wav_np.astype(np.float32))
np.savez(os.path.join(OUT, "stages.npz"), **{k: v.astype(np.float32) for k, v in stages.items()})

# PCM16 WAV so Zero can listen
pcm = (np.clip(wav_np.reshape(-1), -1, 1) * 32767.0).astype(np.int16)
with wave.open(os.path.join(OUT, "ref.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm.tobytes())

def csum(a): return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
meta = {
    "T_frames": T_FRAMES, "seed": SEED, "sample_rate": 24000,
    "codes_shape": list(codes.shape), "wav_shape": list(wav_np.shape),
    "n_samples": int(wav_np.size), "duration_s": round(wav_np.size / 24000, 3),
    "manual_vs_model_maxdiff": agree,
    "checksums": {k: csum(v) for k, v in stages.items()},
    "codes_checksum": csum(codes.detach().cpu().numpy().astype(np.int64)),
    "wav_rms": float(np.sqrt(np.mean(wav_np**2))), "wav_absmax": float(np.max(np.abs(wav_np))),
}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print("META:", json.dumps(meta, indent=2), flush=True)
print("DONE -> " + OUT, flush=True)
