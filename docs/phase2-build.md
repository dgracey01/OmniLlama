# OmniLlama — Phase 2: Build Plan (converter + bridge)

Status: STARTED 2026-10-16. Scope **collapsed** after Phase-1 measurement: the voice runtime already exists in `X:\llamacpp` for Qwen3-TTS. OmniLlama = **adapt that path to the Omni checkpoint**, not build a runtime.

## What already exists in `X:\llamacpp` (reuse — do NOT rebuild)
| Piece | Where | Reuse |
|---|---|---|
| Thinker (MoE text + image/audio INPUT) | `conversion/qwen3vl.py` (registers `Qwen3OmniMoeForConditionalGeneration`), `src/models/qwen3vl.cpp` | ✅ as-is (the existing "mute" GGUF) |
| Talker backbone (AR text model) | arch `QWEN3TTS` → `src/models/qwen3tts.cpp` **reuses `qwen3vl` graph (MoE-capable)** | ✅ graph reusable for Omni's MoE Talker |
| Code Predictor / MTP (15 codebooks) | `MTMD_GEN_PROCESS_TYPE_GEN_CODE`, tensors `A_GEN_CODE_*`, graph in `tools/mtmd/models/qwen3tts-gen.cpp` | ✅ same structure as Omni's `talker.code_predictor.*` |
| **Code2Wav vocoder (FULL)** | `MTMD_GEN_PROCESS_TYPE_GEN_WAV`, tensors `A_GEN_WAV_*`, graph `qwen3tts-gen.cpp::code2wav::decode` (snake/causal_conv1d/_dw/_transpose1d/convnext/dac + streaming state) + `mtmd-audio.cpp` ISTFT | ✅ exact architecture match (pre_transformer SW-attn → 2× ConvNeXt upsample → DAC SnakeBeta → 24 kHz) |
| Converter template | `conversion/qwen3tts.py` (477 lines) — Talker→TextModel (folds text_projection into embd, concats codec vocab), SpeakerEnc+CodePred+Code2Wav→MmprojModel | ✅ subclass + override deltas |

## The Omni-vs-TTS deltas = the ONLY new work
1. **Talker is MoE in Omni** (`talker.model.layers.N.mlp.{experts.N.*, shared_expert.*, gate}`). TTS backbone reuses qwen3vl (MoE-capable) → likely works via converter; **verify the QWEN3TTS hparam path reads `num_experts`/`shared_expert`** (may need a tweak).
2. **code2wav source + naming differ — RESOLVED by Phase 2a (`refs/code2wav_shapes.txt`).** TTS reads a SEPARATE `speech_tokenizer/model.safetensors` (`decoder.*`); **Omni ships `code2wav.*` in the MAIN safetensors.** Full Omni structure now pinned and it maps **1:1** to the existing `A_GEN_WAV_*` graph except for ONE item:
   - `code2wav.pre_transformer.layers.0-7.{self_attn.{q,k,v,o}_proj [1024,1024], mlp.{gate,up,down}_proj (3072), {self_attn,mlp}_layer_scale.scale, input/post layernorm}` + `pre_transformer.norm` → `A_GEN_WAV_TFM_*` (8 layers; NOTE Omni has **no** input_proj/output_proj — TTS does → emit identity or drop).
   - `code2wav.upsample.{0,1}.0.conv [1024,1024,2]` (ConvTranspose) + `.1.{dwconv [1024,1,7], norm, pwconv1 [4096,1024], pwconv2 [1024,4096], gamma}` → `A_GEN_WAV_UP_*` (**1:1**).
   - `code2wav.decoder.0.conv [1536,1024,7]` (entry) → `A_GEN_WAV_DAC_ENTRY`; `decoder.{1..4}.block.{0 snake, 1 convT [k=16,10,8,6], 2/3/4 res units(act1/conv1/act2/conv2)}` → `A_GEN_WAV_DAC_UP_*`/`_RES_*`; `decoder.5.{alpha,beta}` → `A_GEN_WAV_DAC_POST_SNAKE`; `decoder.6.conv (→1ch)` → `A_GEN_WAV_DAC_POST_CONV` (**1:1**; SnakeBeta alpha/beta fold via exp/recip as `qwen3tts.py::snake_fold`).
   - ⚠️ **THE ONE GENUINE VARIANT:** Omni has a **flat `code_embedding [32768,1024]`** (16 codebooks × 2048), NOT TTS's RVQ `quantizer.rvq_first/rvq_rest` (input_proj/codebook/output_proj). `quant_decode()` in `qwen3tts-gen.cpp` does RVQ decode; Omni needs a **small variant**: index the flat table at `cb*2048 + code` for each of 16 codebooks and sum → the 1024-dim frame vector. Converter emits `code_embedding` as one tensor; runtime gets a flat-embed path. **This is the only new C++ in Code2Wav; the rest is reuse.**
3. **`talker.hidden_projection.*`** (multimodal conditioning from Thinker layer-24) exists in Omni, not TTS. Needed only for multimodal INPUT; **text-only works via `text_projection` (token-id handoff) exactly like TTS.** New runtime path for the multimodal case (defer to a later phase; text-first).
4. Codec rate 12.5 Hz (Omni) vs 12 Hz (TTS) — `position_id_per_seconds: 13`, timing params.

## Ordered build steps
- **2a (measure, on Goliath): dump Omni `code2wav.*` tensor names + shapes** and diff against the `A_GEN_WAV_*` graph inputs in `qwen3tts-gen.cpp::code2wav`. Decides whether code2wav maps by rename alone or needs a quantizer-handling tweak. *(script: `src/inspect_code2wav.py`)*
- **2b: Omni voice converter** — `src/omnillama_voice_convert.py` subclasses `conversion/qwen3tts.py`:
  - **Talker**: `Qwen3OmniTalker(Qwen3TTSTalkerModel)` — source prefix `talker.` (already matches); keep MoE experts; register for `Qwen3OmniMoeForConditionalGeneration` (talker side).
  - **Code Predictor**: reuse `Qwen3TTSSpeakerEncoderModel`'s `talker.code_predictor.*` → `A_GEN_CODE_*` mapping (Omni names match: `lm_head.N`, `model.codec_embedding.N`, `model.layers.N.*`, `small_to_mtp_projection`).
  - **Code2Wav**: override `_generate_code2wav_tensors()` to read from MAIN safetensors `code2wav.*` and remap to `A_GEN_WAV_*` per the 2a diff.
- **2c: integrate + run on Goliath** — drop converter into `X:\llamacpp/conversion/`, register in `conversion/__init__.py`, run against `Z:\Qwen3-Omni-30B-A3B-Instruct`, produce the Talker GGUF + the gen/mmproj GGUF(s).
- **2d: build + smoke-test on Goliath (3090/CUDA)** via the bridge: load Thinker (existing qwen3vl GGUF) + the new Talker/CodePred/Code2Wav; feed Thinker token stream → Talker → MTP → Code2Wav → WAV. Text-only path first (no hidden_projection).
- **Phase 3 (later): multimodal conditioning** — wire `talker.hidden_projection` so audio/vision input reaches the Talker. Then quantize for the 580 (IQ3 Thinker + Q8/F16 voice).

## Build environment
- **All build/convert/test runs on Goliath** (`X:\llamacpp`, 3090/CUDA, model local via E:, bridge per [[project_goliath_bridge]]). TheBeast authors (no cmake / no `gguf` pkg in py3.14).
- **Do NOT mutate `X:\llamacpp` blindly.** New converter lives in the OmniLlama project; integration = a documented copy-in + `__init__.py` registration, applied deliberately, tested on Goliath. (The clone is "not a git repo" → no easy revert; treat as precious.)

## Risk notes
- code2wav quantizer reconciliation (step 2a) is the main unknown — but it's a mapping task, graph already exists.
- MoE-talker hparam path in QWEN3TTS may need a small patch.
- `X:\llamacpp` is a non-git working tree → snapshot before integrating; keep OmniLlama changes as copy-in files + a patch list.
