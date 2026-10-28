#!/usr/bin/env python3
r"""OmniLlama 5b-2 gold reference (Goliath) -- ISOLATED talker (no 30B thinker; Goliath has ~11GiB RAM free).

VERIFIED from transformers source (modeling_qwen3_omni_moe.py generate / _get_talker_assistant_parts):
for TEXT-in -> SPEECH-out the talker consumes ONLY text_projection(embed_tokens(ids)) -- the thinker's
hidden states (accept_hidden_layer=24 + hidden_projection) are used ONLY for multimodal user tokens.
So this reference loads ONLY: thinker.model.embed_tokens + the Talker submodule + Code2Wav (~7GiB, fits
24GiB), supplies a FIXED spoken text in place of thinker generation (a controlled yardstick), and
reproduces the pipeline's exact input assembly + talker.generate + code2wav.chunked_decode.

Outputs -> I:\_bridge\talker_ref\ : talker_input_embed_f32.bin/.shape, trailing_text_hidden*, tts_pad_embed*,
talker_input_ids_i64.bin/.shape, codes.npy [1,16,T], talker_ref.wav / ref_wav.npy, meta.json.
These are the exact replay inputs + expected codec grid + waveform for the llama.cpp Talker harness (5b-3).
"""
import os, sys, json, time, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
torch.set_grad_enabled(False)

MODEL   = r"E:\Qwen3-Omni-30B-A3B-Instruct"
OUT     = r"I:\_bridge\talker_ref"
SPEAKER = "Ethan"
PROMPT  = "Please read this aloud."
SPOKEN  = "Hello, this is a test of the Omni voice."
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
DT      = torch.bfloat16
os.makedirs(OUT, exist_ok=True)

import psutil
vm = psutil.virtual_memory()
print(f"[env] torch {torch.__version__} dev={DEV} RAM avail={vm.available/2**30:.1f}/{vm.total/2**30:.1f}GiB", flush=True)

from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration, Qwen3OmniMoeCode2Wav,
)

cfg = AutoConfig.from_pretrained(MODEL)
tcfg, ccfg = cfg.talker_config, cfg.code2wav_config
D = tcfg.text_config.hidden_size
print(f"[cfg] talker hidden={D} layers={tcfg.text_config.num_hidden_layers} "
      f"experts={tcfg.text_config.num_experts} codec_vocab={tcfg.text_config.vocab_size}", flush=True)

# ---- load ONLY talker.* + code2wav.* + thinker embed_tokens from the shards ----
def load_prefixed(prefixes):
    want = {}
    for shard in sorted(glob.glob(os.path.join(MODEL, "model-*-of-*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                for p in prefixes:
                    if k == p or k.startswith(p):
                        want[k] = f.get_tensor(k)
    return want

print("[load] reading talker/code2wav/embed tensors from shards ...", flush=True)
t0 = time.time()
raw = load_prefixed(("talker.", "code2wav.", "thinker.model.embed_tokens.weight"))
print(f"[load] {len(raw)} tensors in {time.time()-t0:.1f}s", flush=True)

talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(tcfg).to(DT)
code2wav = Qwen3OmniMoeCode2Wav._from_config(ccfg).to(DT)

talker_sd   = {k[len("talker."):]:   v for k, v in raw.items() if k.startswith("talker.")}
code2wav_sd = {k[len("code2wav."):]: v for k, v in raw.items() if k.startswith("code2wav.")}

# transformers 5.8 stores talker backbone experts FUSED (experts.gate_up_proj [E,2*inter,hidden] =
# cat[gate;up]; experts.down_proj [E,hidden,inter]); the checkpoint is per-expert. from_pretrained fuses
# via a load hook that our manual load skips -> fuse here or all experts stay random-init (silent, fatal).
nL, nE = tcfg.text_config.num_hidden_layers, tcfg.text_config.num_experts
for lid in range(nL):
    pre = f"model.layers.{lid}.mlp.experts."
    if (pre + "0.gate_proj.weight") not in talker_sd:
        continue
    gu, dn = [], []
    for e in range(nE):
        g = talker_sd.pop(f"{pre}{e}.gate_proj.weight")
        u = talker_sd.pop(f"{pre}{e}.up_proj.weight")
        dn.append(talker_sd.pop(f"{pre}{e}.down_proj.weight"))
        gu.append(torch.cat([g, u], dim=0))          # [2*inter, hidden]
    talker_sd[pre + "gate_up_proj"] = torch.stack(gu, dim=0)  # [E, 2*inter, hidden]
    talker_sd[pre + "down_proj"]    = torch.stack(dn, dim=0)  # [E, hidden, inter]

mt = talker.load_state_dict(talker_sd, strict=False)
mc = code2wav.load_state_dict(code2wav_sd, strict=False)
print(f"[load] talker missing={len(mt.missing_keys)} unexpected={len(mt.unexpected_keys)}; "
      f"code2wav missing={len(mc.missing_keys)} unexpected={len(mc.unexpected_keys)}", flush=True)
for tag, m in (("talker", mt), ("code2wav", mc)):
    if m.missing_keys:    print(f"  [{tag}] MISSING (first 8):", m.missing_keys[:8], flush=True)
    if m.unexpected_keys: print(f"  [{tag}] UNEXPECTED (first 8):", m.unexpected_keys[:8], flush=True)
assert not mt.missing_keys and not mc.missing_keys, "weights did not load cleanly -> reference would be invalid"

embed_w = raw["thinker.model.embed_tokens.weight"].to(DT)
embed = torch.nn.Embedding(embed_w.shape[0], embed_w.shape[1], _weight=embed_w)
talker, code2wav, embed = talker.to(DEV).eval(), code2wav.to(DEV).eval(), embed.to(DEV).eval()

# ---- build the full chatml token sequence with a FIXED assistant turn ----
processor = AutoProcessor.from_pretrained(MODEL)
conv = [{"role": "user",      "content": [{"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": SPOKEN}]}]
text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
ids = processor(text=text, return_tensors="pt")["input_ids"].to(DEV)  # [1, T]
print(f"[seq] full chatml tokens = {ids.shape[1]}", flush=True)

thinker_embed = embed(ids)  # [1 T d]  == hidden_states[0] in the real pipeline

# ---- special embeds (mirror generate() lines 4016-4025) ----
tts_special = torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]], device=DEV)
tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(embed(tts_special)).chunk(3, dim=1)
speaker_id = tcfg.speaker_id[SPEAKER.lower()]

def assistant_parts(s, e):  # mirror _get_talker_assistant_parts (3838-3893)
    ah = talker.text_projection(thinker_embed[:, s:e])  # [1 t d]
    assistant_text = torch.cat((ah[:, :3], tts_pad_embed.expand(-1, 4, -1), tts_bos_embed, ah[:, 3:4]), dim=1)
    codec_special = torch.tensor([[tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id,
                                   speaker_id, tcfg.codec_pad_id, tcfg.codec_bos_id]], device=DEV)
    assistant_codec = torch.cat((torch.zeros((1, 3, D), device=DEV, dtype=talker.dtype),
                                 talker.get_input_embeddings()(codec_special)), dim=1)
    trailing = torch.cat((ah[:, 4:], tts_eos_embed), dim=1)
    inp = assistant_text + assistant_codec
    iid = torch.full((1, assistant_text.shape[1]), cfg.tts_pad_token_id, dtype=torch.long, device=DEV)
    return inp, iid, trailing

# ---- segment loop (mirror generate() 4003-4063), text-only so no multimodal/hidden path ----
im_starts = torch.cat((torch.nonzero(ids[0] == cfg.im_start_token_id).squeeze(-1),
                       torch.tensor([ids.shape[1]], device=DEV)), dim=-1)
embeds, iids, trailing_text_hidden = [], [], None
for i in range(len(im_starts) - 1):
    s, e = int(im_starts[i]), int(im_starts[i + 1])
    role = int(ids[0][s + 1])
    if role == cfg.system_token_id:
        continue
    if role == cfg.user_token_id:
        embeds.append(talker.text_projection(thinker_embed[:, s:e]))   # text-only user part
        iids.append(ids[:, s:e])
    elif role == cfg.assistant_token_id and i == len(im_starts) - 2:
        ae, ai, trailing_text_hidden = assistant_parts(s, e)
        embeds.append(ae); iids.append(ai)
talker_input_embed = torch.cat(embeds, dim=1)
talker_input_id    = torch.cat(iids, dim=1)
print(f"[asm] talker_input_embed={tuple(talker_input_embed.shape)} trailing={tuple(trailing_text_hidden.shape)}", flush=True)

# ---- talker.generate (greedy, deterministic) then code2wav ----
suppressed = [i for i in range(tcfg.text_config.vocab_size - 1024, tcfg.text_config.vocab_size)
              if i != tcfg.codec_eos_token_id]
print("[gen] talker.generate (greedy) ...", flush=True)
# 5b-3 capture: record the EXACT per-step talker inputs_embeds (codec-feedback sum + text stream) that
# prepare_inputs_for_generation builds each decode step. Feeding these to the llama.cpp Talker isolates
# and validates the backbone graph numerics without re-deriving the MTP-coupled codec feedback in C++.
import inspect as _inspect
_captured = []
_orig_prep = talker.prepare_inputs_for_generation
def _cap_prep(*a, **k):
    inp = _orig_prep(*a, **k)
    ie = inp.get("inputs_embeds")
    _captured.append(None if ie is None else ie.detach().float().cpu().numpy())
    return inp
_cap_prep.__signature__ = _inspect.signature(_orig_prep)  # transformers validates kwargs via this sig
talker.prepare_inputs_for_generation = _cap_prep
t0 = time.time()
with torch.no_grad():
    res = talker.generate(
        inputs_embeds=talker_input_embed, trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed, talker_input_ids=talker_input_id,
        max_new_tokens=512, do_sample=False, eos_token_id=tcfg.codec_eos_token_id,
        suppress_tokens=suppressed, output_hidden_states=True, return_dict_in_generate=True,
    )
codes = torch.stack([h[-1] for h in res.hidden_states if h[-1] is not None], dim=1).transpose(1, 2)  # [1 16 T]
print(f"[gen] codes={tuple(codes.shape)} in {time.time()-t0:.1f}s", flush=True)
# dump the captured per-step inputs_embeds: _captured[0] is the [1,P,D] prefill prompt; the rest are the
# [1,1,D] generated-step inputs (one per frame after frame 0). Stack the [1,1,D] ones -> [G, D].
_steps = [c.reshape(-1, D) for c in _captured if c is not None and c.shape[1] == 1]
if _steps:
    _steparr = np.concatenate(_steps, axis=0).astype(np.float32)  # [G, D]
    _steparr.tofile(os.path.join(OUT, "talker_step_embeds_f32.bin"))
    open(os.path.join(OUT, "talker_step_embeds.shape"), "w").write(",".join(map(str, _steparr.shape)))
    print(f"[cap] talker_step_embeds={_steparr.shape} (prefill+steps captured={len(_captured)})", flush=True)
else:
    print(f"[cap] WARN no per-step embeds captured (captured={len(_captured)})", flush=True)
with torch.no_grad():
    wav = code2wav.chunked_decode(codes.to(code2wav.device), chunk_size=300, left_context_size=25)
wav = wav.float().reshape(-1).cpu().numpy().astype(np.float32)
print(f"[gen] wav samples={wav.size} ({wav.size/24000:.2f}s) rms={np.sqrt(np.mean(wav**2)):.4f}", flush=True)

# ---- save ----
def save_f32(name, t):
    a = t.detach().float().cpu().numpy().astype(np.float32); a.tofile(os.path.join(OUT, name + "_f32.bin"))
    open(os.path.join(OUT, name + ".shape"), "w").write(",".join(map(str, a.shape))); return list(a.shape)
shapes = {"talker_input_embed": save_f32("talker_input_embed", talker_input_embed),
          "trailing_text_hidden": save_f32("trailing_text_hidden", trailing_text_hidden),
          "tts_pad_embed": save_f32("tts_pad_embed", tts_pad_embed)}
talker_input_id.to(torch.int64).cpu().numpy().tofile(os.path.join(OUT, "talker_input_ids_i64.bin"))
np.save(os.path.join(OUT, "codes.npy"), codes.to(torch.int64).cpu().numpy())
np.save(os.path.join(OUT, "ref_wav.npy"), wav)
try:
    import soundfile as sf; sf.write(os.path.join(OUT, "talker_ref.wav"), wav, 24000)
except Exception as e:
    print("[warn] wav write failed:", e, flush=True)
meta = {"prompt": PROMPT, "spoken": SPOKEN, "speaker": SPEAKER, "seq_tokens": int(ids.shape[1]),
        "codes_shape": list(codes.shape), "talker_input_shapes": shapes,
        "talker_input_ids_shape": list(talker_input_id.shape),
        "wav_samples": int(wav.size), "wav_seconds": float(wav.size / 24000.0),
        "wav_rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))),
        "talker_missing": len(mt.missing_keys), "talker_unexpected": len(mt.unexpected_keys),
        "transformers": __import__("transformers").__version__}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), indent=2)
print("[done]", OUT, flush=True); print(json.dumps(meta, indent=2), flush=True)
