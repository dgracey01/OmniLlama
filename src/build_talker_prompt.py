#!/usr/bin/env python3
r"""OmniLlama 5b-5 Tier-1 arbitrary-text front-end (Goliath).

Builds the Talker prompt embeds for ANY spoken text and dumps the three files the 5b-4 C++ orchestrator
(llama-talker-tts-test) consumes: talker_input_embed_f32.bin, trailing_text_hidden_f32.bin,
tts_pad_embed_f32.bin. This is exactly the prompt-assembly half of the 5b-2 gold reference
(omni_talker_reference.py lines ~99-142) with the talker.generate + code2wav + capture removed -- those
stages are now done by the fork. So: run this for a sentence, then run the orchestrator on the out dir
and the Omni GGUF speaks that sentence.

Needs only: thinker.model.embed_tokens (text lookup) + talker.text_projection (2048->1024) +
talker.codec_embedding (specials). No code2wav load, no generation here.

usage: python build_talker_prompt.py "<spoken text>" <out_dir> [speaker] [user_prompt]
   e.g. python build_talker_prompt.py "The quick brown fox." I:\_bridge\talker_arb Ethan
"""
import os, sys, json, glob, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
torch.set_grad_enabled(False)

MODEL = r"E:\Qwen3-Omni-30B-A3B-Instruct"   # Goliath E: (= Z: from TheBeast)

SPOKEN  = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of the Omni voice."
OUT     = sys.argv[2] if len(sys.argv) > 2 else r"I:\_bridge\talker_arb"
SPEAKER = sys.argv[3] if len(sys.argv) > 3 else "Ethan"
PROMPT  = sys.argv[4] if len(sys.argv) > 4 else "Please read this aloud."
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
DT      = torch.bfloat16
os.makedirs(OUT, exist_ok=True)
print(f"[args] spoken={SPOKEN!r} speaker={SPEAKER} out={OUT}", flush=True)

from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

cfg = AutoConfig.from_pretrained(MODEL)
tcfg = cfg.talker_config
D = tcfg.text_config.hidden_size
print(f"[cfg] talker hidden={D} codec_vocab={tcfg.text_config.vocab_size}", flush=True)

def load_prefixed(prefixes):
    want = {}
    for shard in sorted(glob.glob(os.path.join(MODEL, "model-*-of-*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if any(k == p or k.startswith(p) for p in prefixes):
                    want[k] = f.get_tensor(k)
    return want

print("[load] reading talker + embed_tokens tensors ...", flush=True)
t0 = time.time()
raw = load_prefixed(("talker.", "thinker.model.embed_tokens.weight"))
print(f"[load] {len(raw)} tensors in {time.time()-t0:.1f}s", flush=True)

talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(tcfg).to(DT)
talker_sd = {k[len("talker."):]: v for k, v in raw.items() if k.startswith("talker.")}

# fuse per-expert MoE weights (from_pretrained hook that a manual load would skip); harmless even though we
# never run the backbone here -- keeps the load assert honest.
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
        gu.append(torch.cat([g, u], dim=0))
    talker_sd[pre + "gate_up_proj"] = torch.stack(gu, dim=0)
    talker_sd[pre + "down_proj"]    = torch.stack(dn, dim=0)

mt = talker.load_state_dict(talker_sd, strict=False)
assert not mt.missing_keys, f"talker weights did not load cleanly: {mt.missing_keys[:8]}"

embed_w = raw["thinker.model.embed_tokens.weight"].to(DT)
embed = torch.nn.Embedding(embed_w.shape[0], embed_w.shape[1], _weight=embed_w)
talker, embed = talker.to(DEV).eval(), embed.to(DEV).eval()

# ---- build chatml, embed, project (identical assembly to the gold reference) ----
processor = AutoProcessor.from_pretrained(MODEL)
conv = [{"role": "user",      "content": [{"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": SPOKEN}]}]
text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
ids = processor(text=text, return_tensors="pt")["input_ids"].to(DEV)
print(f"[seq] full chatml tokens = {ids.shape[1]}", flush=True)
thinker_embed = embed(ids)

tts_special = torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]], device=DEV)
tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(embed(tts_special)).chunk(3, dim=1)
speaker_id = tcfg.speaker_id[SPEAKER.lower()]

def assistant_parts(s, e):
    ah = talker.text_projection(thinker_embed[:, s:e])
    assistant_text = torch.cat((ah[:, :3], tts_pad_embed.expand(-1, 4, -1), tts_bos_embed, ah[:, 3:4]), dim=1)
    codec_special = torch.tensor([[tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id,
                                   speaker_id, tcfg.codec_pad_id, tcfg.codec_bos_id]], device=DEV)
    assistant_codec = torch.cat((torch.zeros((1, 3, D), device=DEV, dtype=talker.dtype),
                                 talker.get_input_embeddings()(codec_special)), dim=1)
    trailing = torch.cat((ah[:, 4:], tts_eos_embed), dim=1)
    inp = assistant_text + assistant_codec
    iid = torch.full((1, assistant_text.shape[1]), cfg.tts_pad_token_id, dtype=torch.long, device=DEV)
    return inp, iid, trailing

im_starts = torch.cat((torch.nonzero(ids[0] == cfg.im_start_token_id).squeeze(-1),
                       torch.tensor([ids.shape[1]], device=DEV)), dim=-1)
embeds, iids, trailing_text_hidden = [], [], None
for i in range(len(im_starts) - 1):
    s, e = int(im_starts[i]), int(im_starts[i + 1])
    role = int(ids[0][s + 1])
    if role == cfg.system_token_id:
        continue
    if role == cfg.user_token_id:
        embeds.append(talker.text_projection(thinker_embed[:, s:e]))
        iids.append(ids[:, s:e])
    elif role == cfg.assistant_token_id and i == len(im_starts) - 2:
        ae, ai, trailing_text_hidden = assistant_parts(s, e)
        embeds.append(ae); iids.append(ai)
talker_input_embed = torch.cat(embeds, dim=1)
talker_input_id    = torch.cat(iids, dim=1)
print(f"[asm] talker_input_embed={tuple(talker_input_embed.shape)} trailing={tuple(trailing_text_hidden.shape)}", flush=True)

def save_f32(name, t):
    a = t.detach().float().cpu().numpy().astype(np.float32); a.tofile(os.path.join(OUT, name + "_f32.bin"))
    open(os.path.join(OUT, name + ".shape"), "w").write(",".join(map(str, a.shape))); return list(a.shape)
shapes = {"talker_input_embed":  save_f32("talker_input_embed", talker_input_embed),
          "trailing_text_hidden": save_f32("trailing_text_hidden", trailing_text_hidden),
          "tts_pad_embed":        save_f32("tts_pad_embed", tts_pad_embed)}
talker_input_id.to(torch.int64).cpu().numpy().tofile(os.path.join(OUT, "talker_input_ids_i64.bin"))
meta = {"spoken": SPOKEN, "prompt": PROMPT, "speaker": SPEAKER, "seq_tokens": int(ids.shape[1]),
        "talker_input_shapes": shapes, "transformers": __import__("transformers").__version__}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), indent=2)
print("[done]", OUT, flush=True); print(json.dumps(meta, indent=2), flush=True)
