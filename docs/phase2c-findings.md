# OmniLlama — Phase 2c START findings (2026-10-27)

Gate = GO (both checks passed, see `gate-checks-complete.md`). Began the Phase 2c runbook against the
live model (`Z:\Qwen3-Omni-30B-A3B-Instruct`) and the newer llama.cpp clone (`X:\llamacpp`, non-git).
This file records what was VERIFIED and the REAL remaining work — correcting the optimistic
"STAGED DROP-IN-READY" status. Nothing here is a guess; it's read from the live config.json, the
safetensors index (28,010 tensors), the TTS converter source, and the transformers Omni modeling code.

## Preconditions (done)
- **3090 free** (0 MiB used, 0% util) — build window open; SeedLora still deferred behind this.
- **Snapshot taken:** `I:\_backup\llamacpp_pre_omnillama` (pristine Sep-22 `qwen3tts.py` + `qwen3tts-gen.cpp`
  verified). This is the ONLY undo for the non-git `X:\llamacpp`.
- **STEP 0 (Windows auto-reboot guard) NOT applied** — the HKLM/WU reg+schtasks changes are blocked by the
  agent safety classifier (reads as weakening Windows Update). Zero must apply STEP 0 by hand in an admin
  PowerShell on Goliath if he wants reboot protection during the build. Build window is ~hours (not the
  month-long SeedLora run), so reboot risk is far lower here.

## Dispatch collision (must solve before converting the voice path)
`conversion/__init__.py` maps each HF arch name → ONE module, and registration is keyed by
`(model_type ∈ {TEXT, MMPROJ}, arch_name)` — only two class slots per arch, picked by `--mmproj`.
`Qwen3OmniMoeForConditionalGeneration` already fills BOTH slots via `qwen3vl`: TEXT = Thinker LLM,
MMPROJ = vision+audio INPUT encoders. There is **no `--arch` override** (`get_model_architecture` reads
`hparams["architectures"][0]`). So the staged `@ModelBase.register(OMNI_ARCH)` for the Talker/VoiceGen can
never fire — `get_model_class` only imports the one mapped module (`qwen3vl`).
**Fix:** convert the voice path from a STAGING DIR whose `config.json` advertises a NEW arch name
(e.g. `Qwen3OmniMoeTalkerForConditionalGeneration`), register the two Omni voice classes under that name
in both maps + the omnillama module, point its safetensors at the real shards (symlink/weight-map copy).
Thinker still converts normally from the original dir. (TTS dodges this only because it has a dedicated
arch and no separate Thinker.)

## Code2Wav — VALIDATED (staged converter mapping is CORRECT)
Full inventory (230 tensors) matches the staged `_map_code2wav` 1:1:
- `code2wav.code_embedding.weight` [32768,1024] = flat 16×2048 (NO `quantizer.rvq_*`).
- `code2wav.pre_transformer.layers.0..7` — attn q/k/v/o (full MHA 1024, no q/k norm) + `self_attn_layer_scale.scale`
  + SwiGLU mlp + `mlp_layer_scale.scale` + input/post layernorm; `pre_transformer.norm.weight` = output norm.
- `code2wav.upsample.0..1` — `.0.conv` (ConvTranspose) + `.1` ConvNeXt (dwconv/norm/pwconv1/pwconv2/gamma).
- `code2wav.decoder.0.conv` entry; `decoder.1..4.block.{0 snake,1 conv,2/3/4 residual}`; `decoder.5.{alpha,beta}`
  post-snake; `decoder.6.conv` post-conv.
⇒ Converter half of Code2Wav is DONE/correct. Only the C++ FLAT-EMBED FRONT-END is new work
(`code2wav_omni_variant.md`: `A_GEN_WAV_CODE_EMBD`, `quant_decode_flat`, skip pre_conv + tfm in/out proj,
nullable loaders, Omni code2wav hparams). This is the single most isolated/demonstrable milestone (5a).

## Talker — staged converter is WRONG here (real redesign needed)
Inventory + transformers modeling code (`modeling_qwen3_omni_moe.py`) show the Omni Talker front-end
differs fundamentally from the TTS Talker:
- **No `talker.model.text_embedding.weight`.** The TTS class builds `TOKEN_EMBD` by folding
  `text_projection` into `model.text_embedding` and concatenating `codec_embedding`. Omni's Talker
  `get_input_embeddings()` returns **`codec_embedding`** (`[3072,1024]`) — its ONLY embedding. So the TTS
  fold keys on a tensor that doesn't exist → `TOKEN_EMBD` never emitted → broken GGUF.
- **Talker vocab (3072) is the CODEC vocab** (codebook-0 codes + codec special ids: codec_bos 2149,
  codec_eos 2150, pad 2148, nothink 2155, think_bos 2156, think_eos 2157), NOT text.
- **`talker.codec_head.weight` [3072,1024]** = output head, **tied** to `codec_embedding`
  (`_tied_weights_keys`). ⇒ map `TOKEN_EMBD = codec_embedding`, `OUTPUT = codec_head`.
- **Text conditioning is Thinker-coupled.** Text tokens are embedded by the **Thinker's** `embed_tokens`
  (vocab 152064, dim 2048) then projected by `talker.text_projection` (MLP 2048→intermediate→1024) and
  fed as `inputs_embeds`. `talker.hidden_projection` (same MLP shape) projects Thinker **hidden states**
  at audio/image/video positions (multimodal; Phase 3). There is NO standalone text table to fold into,
  so the talker can't be a self-contained llama.cpp text LM. ⇒ The C++ orchestrator must, per turn, run
  `thinker.embed_tokens → text_projection` and inject the result as the talker's prompt embeddings, then
  autoregress codec tokens via `codec_embedding`/`codec_head`. This is genuine two-model orchestration,
  bigger than the scoping note's optimistic "token-ids only / clean 2-stage."
- Talker backbone itself = standard Qwen3-MoE (20 layers, 128 experts/6-used, shared_expert +
  shared_expert_gate, q/k norm). MoE graph exists (qwen3vl/qwen3moe path) — reuse OK.

## Code Predictor (MTP) — mostly maps, one delta
`talker.code_predictor.model.layers.0..4` (dense; q/k/v/o + q/k norm + layernorms + mlp) →
TTS `A_GEN_CODE_*`; `model.codec_embedding.0..14` + `lm_head.0..14` (15 each) → stacked
`A_GEN_CODE_EMBD`/`A_GEN_CODE_HEAD`; `model.norm` → `A_GEN_CODE_OUTPUT_NORM`;
`talker.model.codec_embedding.weight` → `A_GEN_CODE_OUT_EMBD`.
**Delta:** Omni has **no `small_to_mtp_projection`** (TTS → `A_GEN_CODE_PROJ_IN`). The predictor instead
concatenates the talker's last hidden with the codebook-0 embedding directly (dims already match at 1024).
⇒ Converter must not require PROJ_IN; C++ gen-code path must tolerate its absence (verify the runtime
doesn't hard-require it).

## VoiceGen converter class — __init__ needs Omni schema
Staged `Qwen3OmniVoiceGenModel(Qwen3TTSSpeakerEncoderModel)` does NOT override `__init__`, so it hits:
- `hparams["talker_config"]["hidden_size"]` → KeyError (Omni nests it under `talker_config["text_config"]`).
- `hparams["speaker_encoder_config"]["n_layers"]` → KeyError (**Omni has NO speaker-clone encoder**).
- `set_gguf_parameters` writes speaker/ECAPA audio params that don't exist for Omni.
⇒ Rewrite `__init__`/`set_gguf_parameters`: drop the speaker encoder entirely
(`has_audio_encoder=False`), read code-predictor hparams from `talker_config["code_predictor_config"]`,
and feed Code2Wav hparams (from `code2wav_config`) + the new flat-embed flags.

## Code2Wav decode — PINNED from transformers source + GOLD REFERENCE generated (2026-10-27)
Read the authoritative `Qwen3OmniMoeCode2Wav.forward` (transformers 5.8, installed on Goliath's
LLaMA-Factory venv — which ships the FULL Omni impl incl. `Qwen3OmniMoeCode2Wav`). Exact chain:
```
hidden = code_embedding(codes + code_offset).mean(1)   # codes [B,16,T]; code_offset = arange(16)*2048
hidden = pre_transformer(inputs_embeds=hidden)         # 8 layers, ALL sliding_attention win=72, RoPE theta=10000, RMSNorm, SwiGLU, layer_scale
hidden = hidden.permute(0,2,1)                          # [B,1024,T]
for [CausalTransConv(ratio,ratio), ConvNeXtBlock] in upsample(2x [2,2]): ...
for block in decoder: CausalConv(1024->1536,k7) · 4x DecoderBlock([SnakeBeta, CausalTransConv(up_rate), 3x ResidualUnit dil 1/3/9]) · SnakeBeta · CausalConv(->1,k7)
wav = wav.clamp(-1,1)                                   # 24 kHz
```
**C++ SPEC CORRECTIONS vs `code2wav_omni_variant.md`:**
- **COMBINE IS `.mean(1)` (÷16), NOT SUM.** The staged `quant_decode_flat` accumulates with `ggml_add`
  (sum) — must divide by num_codebooks (16) after summing, or the signal is 16x off into the transformer.
- Offset `g*2048` uniform — CONFIRMED (semantic_codebook_size 4096 is encoder-side, NOT the flat table;
  flat `code_embedding` = [32768,1024] = 16x2048).
- pre_transformer = causal **sliding-window-72** on ALL 8 layers + RoPE(theta=10000) + RMSNorm. The C++
  reuse of the TTS `qwen3tts-gen.cpp` pre_transformer must apply SWA-72 causal + RoPE (verify the TTS path
  already does; if it's non-causal/no-RoPE that's added work).
- No pre_conv, no tfm in/out proj (native 1024) — CONFIRMED.

**GOLD REFERENCE (5a yardstick) generated on 3090** — `src/omni_c2w_reference.py` →
`I:\_bridge\c2w_ref\` (= `X:\_bridge\c2w_ref\`): `codes.npy [1,16,64]`, `ref_wav.npy`/`ref.wav`
(5.1 s, 24 kHz, RMS 0.109, listenable), `stages.npz` (after_embed_mean / after_pretfm / after_upsample /
final_wav for STAGE-BY-STAGE C++ debugging), `meta.json` (checksums; seed 1234). All 230 tensors loaded
clean (0 missing/0 unexpected); manual taps == library forward (maxdiff 0.0). This is the exact input +
expected output the llama.cpp code2wav path must reproduce.

## 5a GGUF HALF DONE + VERIFIED (2026-10-27) — converter + constants built a correct mmproj
The voice mmproj GGUF now BUILDS and is structurally verified (first convert, 0 errors, 14 s on Goliath).
- **Converter:** `conversion/omnillama_voice.py` (`Qwen3OmniVoiceGenModel`) + driver `src/convert_omni_voice.py`
  (direct-instantiation — sidesteps the dispatch collision; no staging dir / symlinks / __init__.py edit).
  Fixes the staged `__init__`/`set_gguf_parameters`/`get_audio_config` KeyErrors: reads talker hidden from
  `talker_config["text_config"]["hidden_size"]` (1024), injects a dummy `audio_config` to satisfy
  MmprojModel.__init__, writes ONLY the gen-audio encoder (no ECAPA speaker encoder). Code predictor +
  codec_embedding inherited from the TTS parent (names match; `bid` derivation worked).
- **constants.py:** added `A_GEN_WAV_CODE_EMBD` = `a.gen.wav.code_embd` (enum + TENSOR_NAMES + projector list).
- **Output:** `I:\_bridge\omni_voice_out\mmproj-Qwen3-Omni-30b-Instruct-F16.gguf` (807 MB, 289 tensors).
  Verified (`src/inspect_omni_voice_gguf.py`): `a.gen.wav.code_embd.weight` ne **[1024,32768]** (16x2048 flat);
  **NO** quant.first/rest/pre_conv/tfm-in/out-proj (flat path); tfm blk 0..7 (q/k/v/o all [1024,1024] = full
  MHA 16x64, + ls1/ls2 layer-scale, output_norm); up blk 0..1; dac blk 0..3 (conv k=16/10/8/6 -> up-rates
  [8,5,4,3]); post_snake+post_conv(->1ch); code.head/embd [1024,2048,15], out_embd [1024,3072].
  KVs: `clip.gen.audio.projector_type=qwen3tts_gen`, projection/embedding_length=1024, ffn=3072, block=5,
  head=16, head_kv=8, eps=1e-6.
- **CONFIRMED for the C++ half:** code2wav tfm is **full MHA (kv=16)** -> the TTS hardcode (wav_tfm_n_head=16,
  n_head_kv=16) already matches; no GQA. No in/out proj -> tfm hidden stays **1024** (TTS wav_tfm_n_embd=512 is
  post-in_proj and is NOT used in the graph math — d_head derives from q_w->ne[1]/n_head=64). Only NEW C++:
  flat `quant_decode_flat` (16x get_rows at +g*2048, **mean/16**) + skip pre_conv + skip tfm in/out proj +
  branch on presence of `c2w.code_embd`.

## 5a COMPLETE + VERIFIED on 3090 (2026-10-27) — llama.cpp Code2Wav == transformers reference
The C++ runtime half built and PASSED. Edits (all in the snapshotted `X:\llamacpp`, mirrored to `src/`):
- `clip-model.h`: `clip_code2wav::code_embd` + `clip_hparams::{wav_flat_code_embed,wav_n_codebooks,wav_codebook_size}`.
- `clip-impl.h`: `TN_A_GEN_WAV_CODE_EMBD`.
- `clip.cpp`: loader branches on presence of `code_embd` (flat: load it, RVQ/pre_conv/tfm-proj optional,
  derive n_codebooks=16/codebook_size=2048/tfm_n_embd=1024 from shape); GEN_WAV marshalling checks codes
  vs codebook_size and pre-offsets `+g*2048` host-side.
- `qwen3tts-gen.cpp`: `quant_decode_flat` (16x get_rows, **ggml_scale 1/16 = mean**); `decode()` front-end
  branch (flat skips pre_conv + tfm in/out proj, tfm runs native 1024); `list_c2w_state_slots` guard no
  longer keys on pre_conv_w alone (flat has none) and skips the pre_conv state slot.
- `mtmd-c2w-test.cpp` (+ CMake target `llama-mtmd-c2w-test`): standalone harness, feeds fixed frame-major
  codes straight into `clip_encode` GEN_WAV (no talker/llama), writes WAV + raw f32. Mirrors mtmd's own
  GEN_WAV call (dummy 1x1 image placeholder, cold-start single window).

BUILD: VS2026-Insiders cmake/ninja + CUDA 12.9 (`-allow-unsupported-compiler`), arch 86; configure + build
both exit 0 FIRST TRY (`I:\llamacpp\build\bin\llama-mtmd-c2w-test.exe`).
RESULT (`c2w_prep_and_compare.py compare`): out 122880 samples vs ref 122325; **best lag −555 samples
(−23 ms) corr=0.99826, rms_err=0.0064 (5.9% of ref RMS)**; out_rms 0.1095 ≈ ref_rms 0.1094. Zero-lag
corr≈0 was purely the causal-conv startup-latency shift (122880−122325 = 555, the warmup prefix the
one-shot transformers forward trims). Artifacts: `I:\_bridge\c2w_ref\out.wav` (llama.cpp audio) vs `ref.wav`.
=> The single new C++ (flat-embed Code2Wav) is CORRECT. 5a (Code2Wav in isolation) DONE.

## 5b DESIGN PINNED (2026-10-27) — authoritative, read from config.json + safetensors index + fork graph source
Zero's plan: character LLMs → voice-enabled Omni GGUFs. Every route (C decoupled-TTS / A Omni+char-LoRA /
B graft) needs the BASE Omni voice path running first = 5b. So 5b proceeds regardless of the C/A/B fork.

**KEY SIMPLIFICATION (text path folds — no live two-model coupling for 5b):** The TTS converter folds
`text_projection ∘ text_embedding` into `TOKEN_EMBD` and concatenates `codec_embedding`. Omni's ONLY
difference is the *text* embedding table lives in the **Thinker** (`thinker.model.embed_tokens [152064,2048]`),
not the Talker. But `talker.text_projection` (linear_fc1/fc2, act silu — VERIFY act in modeling src) is
applied to exactly those embeddings, so fold `text_projection(thinker.embed_tokens)` → `[152064,1024]` text
table + concat `codec_embedding [3072,1024]` → `TOKEN_EMBD [155136,1024]`; `OUTPUT = codec_head [3072,1024]`
(logit-pad like TTS). ⇒ text handoff = TOKEN-IDS (matches paper). The un-foldable `talker.hidden_projection`
(multimodal hidden states) is Phase-3 ONLY, not needed for text→speech.

**AUTHORITATIVE Talker schema (Z:\...\config.json talker_config.text_config + index):**
- backbone = Qwen3-MoE: hidden 1024, 20 layers, 16 heads × head_dim **128** (attn dim 2048), GQA **kv=2**
  (256), q_norm/k_norm, rope θ **1e6**, rms_eps 1e-6, NO sliding window. vocab 3072 (codec).
- FFN = **Qwen2-MoE style**: 128 experts (moe_intermediate_size **384**), num_experts_per_tok 6,
  `mlp.gate` router, **`mlp.shared_expert` (768)** + **`mlp.shared_expert_gate`**. (plain qwen3moe has NO
  shared expert — so the talker is a HYBRID.)
- tensors: `talker.model.codec_embedding.weight [3072,1024]` (input), `talker.codec_head.weight [3072,1024]`
  (output, tied), `talker.model.layers.{0..19}.{self_attn.q/k/v/o_proj, q_norm, k_norm, input_layernorm,
  post_attention_layernorm, mlp.experts.{0..127}.{gate,up,down}_proj, mlp.gate, mlp.shared_expert.{gate,up,
  down}_proj, mlp.shared_expert_gate}`, `talker.model.norm.weight`, `talker.text_projection.linear_fc1/fc2
  (+bias)`, `talker.hidden_projection.linear_fc1/fc2 (+bias)` [skip]. codec ids: bos 2149, eos 2150, pad 2148,
  nothink 2155, think_bos 2156, think_eos 2157. num_code_groups 16, accept_hidden_layer 24.

**CRITICAL C++ FINDING (fork graph source):** `src/models/qwen3tts.cpp` is empty — QWEN3TTS REUSES
`llama_model_qwen3vl` (`qwen3vl.cpp`), whose `load_arch_tensors` builds a **DENSE** FFN (`ffn_gate/up/down`
only; NO experts, NO shared_expert) + the codec logit-pad (`if arch==QWEN3TTS: n_vocab_out=3072`, line 20-23).
TTS-1.7B talker is DENSE. ⇒ Converting the Omni MoE talker under QWEN3TTS would IGNORE all experts = broken.
MoE graphs DO exist to copy: `qwen3vlmoe.cpp` (qwen3 attn + MoE) and `qwen2moe`-style shared-expert FFN.
⇒ **5b-1 runtime = a small NEW arch** (e.g. `QWEN3OMNITALKER`): qwen3vlmoe attention (q/k-norm, GQA) +
qwen2moe shared-expert FFN + QWEN3TTS codec logit-pad (n_vocab_out 3072 + suppress_tokens). Needs: gguf-py
MODEL_ARCH + MODEL_TENSORS list (constants.py), llama-arch.cpp/.h registration + tensor-name map, a graph
class (copy qwen3vlmoe, add shared_expert from qwen2moe, add logit-pad from qwen3vl QWEN3TTS branch). Bounded
C++, testable on Goliath.

**5b component split (end-to-end text→WAV):**
1. **Thinker** GGUF (qwen3vlmoe text LM) — generates response WORDS. [stock convert; already works]
2. **Talker backbone** GGUF (NEW qwen3-omni-talker arch) — re-embeds thinker text tokens via the folded
   TOKEN_EMBD, autoregresses codebook-0 codec tokens. [5b-1 = converter + arch + graph; NEW]
3. **mmproj** GGUF (A_GEN_CODE_* MTP + A_GEN_WAV_* code2wav) — cb0→cb1..15 (MTP), 16 codebooks→WAV.
   [code2wav DONE+VERIFIED in 5a; MTP tensors already converted in the 5a mmproj, MTP runtime untested]
   NB: for text→speech the gold-ref can SKIP the 30B thinker forward entirely — the text path only needs
   `embed_tokens` (a table) + talker + MTP + code2wav (~few GB, fits 24GB), feeding FIXED text tokens.

## 5b sub-milestones (de-risk order, each a demonstrable gate)
- **5b-2 (do FIRST — pure Python, no new C++):** Talker gold reference. Load embed_tokens table + talker +
  code_predictor + code2wav via transformers (NOT the 30B thinker), feed a FIXED short text, capture
  cb0 sequence + MTP cb1..15 → codec grid [1,16,T] + final WAV. The yardstick, like `omni_c2w_reference.py`.
- **5b-1:** Talker converter (`omnillama_talker.py`, direct-instantiation driver) → structural GGUF dump
  verify; + the NEW arch (gguf-py + llama-arch + graph) → build on Goliath.
- **5b-3:** Talker-in-isolation C++ harness: inject folded text embeds (or feed text ids), autoregress cb0,
  run MTP → compare codec grid to 5b-2 gold; feed to 5a code2wav → coherent WAV.
- **5b-4:** Full orchestrator: Thinker → Talker → MTP → Code2Wav in one process → text → WAV end-to-end.
- then: Thinker imatrix IQ3 for the 580; keep voice F16/Q8.

## 5b-2 COMPLETE + VALID on 3090 (2026-10-27) — talker gold reference generated
`src/omni_talker_reference.py` (run on Goliath via `I:\_bridge\talker_ref_run.bat`) → `I:\_bridge\talker_ref\`
(= `X:\_bridge\talker_ref\`). ISOLATED talker (NO 30B thinker — **Goliath has only 15.9 GiB total RAM**,
so the full pipeline is infeasible; the text path needs only embed_tokens + talker + code2wav ≈ 7 GiB on
the 3090). Loads talker.*/code2wav.*/thinker.embed_tokens from the shards, supplies a FIXED spoken line
("Hello, this is a test of the Omni voice."), and reproduces the pipeline's exact input assembly
(_get_talker_assistant_parts) + `talker.generate(greedy)` + `code2wav.chunked_decode(300,25)`.
- **GOTCHA (silent+fatal, fixed):** transformers 5.8 stores the talker backbone experts FUSED
  (`model.layers.{L}.mlp.experts.gate_up_proj [128,768,1024]` = stack of cat[gate(384,1024);up(384,1024)];
  `experts.down_proj [128,1024,384]`). The checkpoint is PER-EXPERT (`experts.{e}.{gate,up,down}_proj.weight`);
  `from_pretrained` fuses via a load hook that a manual `load_state_dict` SKIPS → first run left all
  128×20 experts random-init (load reported missing=40/unexpected=7680), output was noise (rms 0.0012,
  ran to the 512 cap, no EOS). Fix = fuse before load; added a hard assert `missing==0`.
- **VALID RESULT:** clean load (missing=0/unexpected=0); talker emitted a **clean codec_eos at 33 frames**
  (codes `[1,16,33]`), `code2wav` → **2.62 s / rms 0.0606** coherent speech (33×1920≈63k samples). The
  `rope_parameters` 'mrope_section/interleaved' warning is BENIGN (clean EOS + coherent speech prove the
  talker ran right for the 1D text+codec path).
- **Replay artifacts for the C++ harness (5b-3):** `talker_input_embed_f32.bin` [1,19,1024] (the summed
  text+codec prompt stream), `trailing_text_hidden_f32.bin` [1,13,1024], `tts_pad_embed_f32.bin` [1,1,1024],
  `talker_input_ids_i64.bin` [1,19], `codes.npy` [1,16,33] (expected grid), `talker_ref.wav`/`ref_wav.npy`
  (expected 24 kHz waveform), `meta.json`. 5b-3 feeds the embeds in, autoregresses cb0, runs MTP, and
  diffs the codec grid vs codes.npy (and the WAV vs talker_ref.wav via the proven 5a code2wav).

## 5b-1 CONVERTER HALF DONE + VERIFIED on Goliath (2026-10-27)
Talker backbone → GGUF, structurally verified (ALL CHECKS PASS; convert+inspect exit 0, 161 s).
- gguf-py `constants.py`: new `MODEL_ARCH.QWEN3OMNITALKER` (name `qwen3omnitalker`) + MODEL_TENSORS =
  QWEN2MOE shared-expert set + `ATTN_Q_NORM`/`ATTN_K_NORM`.
- `conversion/omnillama_talker.py` (`Qwen3OmniTalkerModel(Qwen3MoeModel)`) + driver `src/convert_omni_talker.py`
  (direct-instantiation). Feeds the talker `text_config` as hparams; strips `talker.` prefix; drops
  text_projection/hidden_projection/code_predictor; `codec_embedding`→TOKEN_EMBD, `codec_head`→OUTPUT;
  codec-only vocab (tokenizer `none`, 3072 CONTROL). Expert 3D-stacking + shared-expert mapping + q/k-norm
  all inherited from Qwen3MoeModel/Qwen2MoeModel/tensor_mapping.
- OUTPUT `I:\_bridge\omni_talker_out\Qwen3-Omni-A3B-30b-Instruct-F16.gguf` (6.3 GB, 323 tensors).
  Verified: token_embd/output [1024,3072]; 20 blk; attn q[1024,2048] k/v[1024,256] (GQA kv2) q/k_norm[128]
  head_dim128; ffn_gate/up_exps[1024,384,128] down[384,1024,128] ×128; shared_expert[1024,768]+gate[1024,1];
  KVs expert_count128/used6, expert_ff384, shared_ff768, rope θ1e6, eps1e-6.
- **C++ graph note:** converter wrote `rope.dimension_sections=[24,20,20,0]` (mrope, inherited from qwen3).
  The gold ref ran with mrope → the talker graph must use **qwen3vlmoe-style `ggml_rope_multi`** + load
  rope_sections (NOT qwen2moe plain rope_ext). "Unknown RoPE type: default" warning is benign.

## 5b-1 C++ HALF COMPLETE + VERIFIED on Goliath 3090 (2026-10-27)
The `qwen3omnitalker` runtime builds, loads the GGUF, builds the CUDA graph, and produces well-formed
3072-wide codec logits. Load+decode smoke test PASSED:
`LOADED n_vocab=3072 → DECODE OK argmax=2150 (codec EOS) max_logit=10.20 mean=-3.43 → TALKER_LOAD_TEST_PASS`.
(argmax=EOS is meaningless on arbitrary input ids — the point is the logits are peaked/sane, not NaN/flat.)

**Files changed (all in the fork I:\llamacpp):**
- `src/llama-arch.h`: `LLM_ARCH_QWEN3OMNITALKER` enum (after POCKETTTS).
- `src/llama-arch.cpp`: arch-name map entry; added to `llm_arch_supports_sm_tensor` false-list next to QWEN3TTS.
  NOTE: this fork uses a **single GLOBAL tensor-name table** (no per-arch LLM_TENSOR_* lists like mainline),
  so no per-arch tensor map was needed here.
- `src/llama-model.cpp`: create case → `llama_model_qwen3omnitalker`; **get_rope_type → NEOX** (NOT imrope —
  see below); n_ff_exp/shexp info logging.
- `src/models/models.h`: `llama_model_qwen3omnitalker` struct.
- `src/models/qwen3omnitalker.cpp` (NEW, auto-globbed): load = qwen3vlmoe tensors (q/k-norm, head_dim-explicit
  QKV) + qwen2moe shared-expert tensors; graph = qwen3vlmoe attention with **`ggml_rope_ext` (NEOX, 1D)** +
  `build_moe_ffn(..., norm_topk_prob=true)` + qwen2moe sigmoid-gated shared-expert block.
- `tools/mtmd/talker-test.cpp` (NEW) + CMake target `llama-talker-test`: minimal load+decode harness.
- `conversion/omnillama_talker.py`: **added `add_vocab_size(3072)`** (see KEY FIX).

**KEY DECISIONS / FIXES (vs the original plan, all verified):**
1. **RoPE = plain NEOX 1D, not mrope.** The gold ref logs "Unknown RoPE type: default" and still makes coherent
   speech ⇒ the Talker uses HF "default" (1D) rope over head_dim 128. So the graph uses `ggml_rope_ext`
   (not `ggml_rope_multi`); the `rope.dimension_sections=[24,20,20,0]` in the GGUF is harmless unused metadata.
   n_rot auto-resolves to 128 (= n_embd_head_k; no rope_dimension_count written).
2. **`norm_topk_prob = true`** — confirmed from config.json talker text_config (NOT qwen2moe's false). Router
   normalizes top-k like qwen3vlmoe.
3. **vocab_size KV is REQUIRED.** tokenizer.model="none" ⇒ `vocab.n_tokens()`=0 UNLESS the GGUF carries
   `{arch}.vocab_size`, which the "none" branch reads to make N dummy tokens (llama-vocab.cpp:1963). Without
   it, token_embd/output/logits all collapse to width 0 and load fails
   (`token_embd.weight expected {1024,0}`). Fix = `add_vocab_size(3072)` in the converter's set_vocab. The C++
   `load_arch_tensors` then uses the macro's `n_vocab` (=3072) unchanged — no logit-pad needed (unlike QWEN3TTS).
4. Build target names changed in this fork (no `llama-cli`; it's `llama-completion`/`llama.exe`). GLOB in
   src/CMakeLists.txt picks up new models/*.cpp on reconfigure.

## 5b-3 COMPLETE + VERIFIED on Goliath 3090 (2026-10-28) — Talker backbone is numerically faithful
Teacher-forced the `qwen3omnitalker` GGUF against the 5b-2 gold reference and checked codebook-0 argmax
per frame. Result: **34/35 frames exact**; the one miss (frame 17: pred 847 vs ref 1899) is a genuine
**F16-vs-bf16 near-tie** — pred_logit 23.266 vs ref_logit 23.094, **gap 0.1719** out of 2048 candidates —
and does **not propagate** (frames 18–34 all exact) because teacher forcing re-feeds the reference's real
inputs each step. ⇒ the ported Talker graph reproduces transformers' Qwen3-Omni Talker to F16 precision.

Harness = `tools/mtmd/talker-embed-test.cpp` (target `llama-talker-embed-test`); it loads the GGUF, feeds
the prompt + the captured per-step inputs via `batch.embd`, and compares argmax to `ref_cb0_i32.bin`.

**Key discovery that shaped the test — the per-step Talker input is NOT `codec_embed(cb0)+text`.** From
`modeling_qwen3_omni_moe.py` `prepare_inputs_for_generation` (lines 3243–3280), each decode step's
`inputs_embeds` = **sum over all 16 codebooks of the PREVIOUS frame** + the text stream:
  - cb0  → `talker.model.codec_embedding(prev_cb0_token)`                         (1 vector)
  - cb1..cb14 → the **MTP (`code_predictor`) per-step last-layer hidden states**  (14 vectors)
  - cb15 → `code_predictor.get_input_embeddings()[-1](cb15_token)`                (1 vector)
  - `+ trailing_text_hidden[:, step]` while `step < R`, else `+ tts_pad_embed`
So Talker and MTP are **coupled** during generation (not separable). The MTP also samples internally
(`do_sample=True`), so cb1..15 — and hence cb0's trajectory AND the sequence length — vary run-to-run
(33 frames one run, 35 the next). That is why 5b-3 compares the Talker against the SAME run's own captured
inputs+codes (the reference dumps `talker_step_embeds_f32.bin` [G,1024] and re-derives `ref_cb0` from the
current `codes.npy`), rather than against a frozen artifact.

Artifacts (regenerated together by `omni_talker_reference.py`, which now patches
`talker.prepare_inputs_for_generation` with a signature-preserving wrapper to capture the per-step embeds):
`talker_input_embed_f32.bin` [1,19,1024], `talker_step_embeds_f32.bin` [G,1024], `codes.npy` [1,16,G+1],
`ref_cb0_i32.bin` (dumped by `dump_talker_ref_aux.py` with `token_embd_f32.bin`).

## 5b-4 COMPLETE + RUNS on Goliath 3090 (2026-10-28) — an Omni GGUF speaks end-to-end in llama.cpp
**Text(fixed)→WAV works.** `tools/mtmd/talker-tts-test.cpp` (target `llama-talker-tts-test`) ties the
verified pieces: Talker GGUF (llama, arch qwen3omnitalker) drives the per-frame loop; the voice mmproj
(clip/mtmd) supplies MTP + Code2Wav via the existing GEN_CODE / GEN_WAV processes. RESULT: loads both,
`gen_audio_type=QWEN3TTS` 24 kHz, generates **32 frames, clean codec_eos stop, 61440 samples = 2.56 s,
rms 0.079** (reference: ~2.6 s, rms 0.061 — same band). cb0 trajectory `1049,196,1914,1239,…` matches the
gold reference EXACTLY for the first frames (same prompt + greedy cb0), then diverges as the seeded MTP
sampling differs from the reference's — expected, still coherent Omni speech from the same prompt.
WAV at `I:\_bridge\talker_ref\omnillama_5b4_out.wav` (+ `.pcm_f32.bin`) for audible check.

**HUGE de-risk discovered during build:** the fork ALREADY implements the whole per-frame orchestration
for Qwen3-TTS, reusable as-is for Omni:
- `tools/mtmd/models/qwen3tts-gen.cpp` `build()` GEN_CODE: takes talker hidden (`inp_raw`) + cb0
  (`inp_code0`), runs the 5-layer code predictor (prefill+step, KV cache) → **output 1** = 16 codes;
  **output 2 `gen_audio_out`** = cb0 embed + sum of 15 residual-codebook embeds = the EXACT talker
  feedback (`codec_hiddens.sum(1)` minus text). `small_to_mtp_projection` is nullable → skipped (Omni).
- `tools/mtmd/mtmd-helper-gen.cpp` `qwen3tts_gen_audio_pipeline` is the full loop (prompt→sample cb0→
  GEN_CODE→overlay→feed back→GEN_WAV windows). Omni deltas vs TTS: (1) prompt = thinker embed_tokens→
  text_projection (not in talker GGUF) — SIDESTEPPED here by replaying the 5b-2 saved prompt embeds;
  (2) overlay is a CHANGING text stream (`trailing_text_hidden[frame]` then `tts_pad`) vs TTS's constant
  pad. The standalone harness drives GEN_CODE/GEN_WAV via `mtmd_gen_audio_process` with the correct overlay.
- Voice mmproj carries the complete MTP: `a.gen.code.blk.0..4` (5 dense blocks, qk-norm, GQA 16/8,
  SwiGLU), `a.gen.code.embd [1024,2048,15]`, `a.gen.code.head [1024,2048,15]`, `a.gen.code.out_embd
  [1024,3072]` (cb0 embed), `output_norm`.

**TWO mtmd FORK FIXES (commit so an engine pull can't revert — see [[project_trainerxl_resume_fix]] lesson):**
`tools/mtmd/mtmd.cpp` rejected a gen-audio-only mmproj (Omni voice stack has NO input/speaker encoder;
5a dropped ECAPA): (1) load guard `if(!ctx_v && !ctx_a)` → `if(!ctx_v && !ctx_a && !ctx_gen_a)`;
(2) the input-encoder n_embd check `clip_n_mmproj_embd(ctx_v?ctx_v:ctx_a)` dereferenced null when both
null — wrapped in `if (ctx_v || ctx_a)` (gen-audio n_embd is validated separately just below).

**Talker context:** `cp.embeddings=true` + `LLAMA_POOLING_TYPE_NONE` so one decode yields BOTH cb0 logits
(sample) and the last hidden (`llama_get_embeddings_ith`, = transformers `past_hidden`, fed to GEN_CODE).
Talker is NEOX → `n_pos_per_embd=1`, `set_position_normal`.

## NEXT — 5b-5 arbitrary-text front-end + audible/quant validation
5b-4 used the saved fixed-sentence prompt embeds. 5b-5 builds the prompt from ARBITRARY text: thinker
`embed_tokens(text_ids)` → talker `text_projection` (an MLP, 2048→1024) + the Omni special-token layout
(assistant_parts), so any string can be spoken. text_projection is NOT in the talker GGUF — options:
(a) add it to the voice mmproj (new `a.gen.*` tensors) and apply orchestrator-side, or (b) a tiny
thinker-embed+projection GGUF. Also: AUDIBLE verification of `omnillama_5b4_out.wav` (Zero), and then the
quant arc (Thinker imatrix IQ3 for 16 GB; voice stack F16/Q8; Q3 stress test) + Tifa character distill.

## (superseded) earlier 5b-1 plan
Converter `conversion/omnillama_talker.py` (direct-instantiation driver like 5a): TOKEN_EMBD =
cat[ text_projection(thinker.embed_tokens) [152064,1024] | codec_embedding [3072,1024] ] (fold act =
VERIFY talker ResizeMLP act, silu expected); OUTPUT = codec_head [3072,1024]; MoE layer tensors via the
standard stacked-expert mapping; suppress_tokens/eos from TTS pattern; skip code_predictor/code2wav
(already in the 5a mmproj) + hidden_projection. New arch `QWEN3OMNITALKER` = qwen3vlmoe attention
(q/k-norm, GQA kv=2) + qwen2moe shared-expert FFN + QWEN3TTS codec logit-pad (n_vocab_out 3072). gguf-py
MODEL_ARCH+MODEL_TENSORS, llama-arch.cpp/.h, graph class. Build on Goliath (1-3 rounds). Then 5b-3 harness.

## Remaining work beyond 5a (decomposed, de-risk order)
1. **Staging-dir arch override** so the voice classes are dispatchable without colliding with the Thinker.
2. **Rewrite the two Omni converter classes** for the real schema (Talker: TOKEN_EMBD=codec_embedding,
   OUTPUT=codec_head, keep text_projection/hidden_projection as side tensors for the orchestrator;
   VoiceGen: drop speaker encoder, wire code2wav hparams). Code2Wav tensor map already correct.
3. **C++ Code2Wav flat-embed variant** (`code2wav_omni_variant.md`) + hparam plumbing + nullable loaders.
4. **Build** llama.cpp CUDA on Goliath (1–3 compile rounds expected).
5. **5a — Code2Wav in isolation** (the one new C++): feed a reference 16×T codec-token sequence → WAV;
   compare to a transformers reference WAV. Single most informative result; prove this BEFORE the talker.
6. **Talker orchestration** (thinker-embed → text_projection → talker prompt embeds → codec AR) — the
   genuinely hard C++. Then **5b end-to-end** text → WAV.
7. Quantize Thinker (imatrix IQ3) for the 580; keep voice F16/Q8; re-test.

## Honest verdict on effort
Consistent with the original "1.5–3 months for a ggml-fluent dev; staged coding-agent build" — NOT a
"run the staged converter and get a WAV today" task. The 2026-10-16 "DROP-IN-READY" label was optimistic:
Code2Wav mapping is genuinely ready, but Talker + VoiceGen + orchestration are real builds. The most
valuable near-term, self-contained deliverable is **5a (Code2Wav WAV in isolation)**; it occupies the
3090 (converts + builds + a transformers reference run) and thus trades against the deferred SeedLora
resume — a scheduling call for Zero.

## CHARACTER VOICE-TRAINING ROUTES (researched 2026-10-27; deferred fork — base 5b finishes first)
Goal: give a character (e.g. Tifa) its own voice inside the Omni GGUF. Researched how, since Omni's vocoder
(Code2Wav) + code-predictor (MTP) are CONFIRMED the same components as Qwen3-TTS (so the TTS training
literature applies directly).

**The recipe (from Qwen3-TTS, Baseten writeup — confirmed):** a voice = supervised fine-tune of the
**Talker (codebook 0) + MTP (codebooks 1–15)**, cross-entropy on `(text, audio)` pairs. Speaker encoder,
codec tokenizer, and vocoder stay **FROZEN**. Voice identity is baked in as a **speaker centroid**
(avg of ~64 ref clips). Cheap: ~1.5 h audio, ~1 h on one H100/3090, LR ≤ 1e-5, MTP-loss weight ~0.3.
Omni holds a voice as a **discrete speaker-id slot** (config talker `speaker_id: {chelsie,ethan,aiden}`),
so a new character = a trained-in speaker slot.

**THE BLOCKER (confirmed from the 28,010-tensor index):** Omni ships an audio *understanding* encoder
(`thinker.audio_tower.*`, mel→semantic features) and the codec *decoder* (`code2wav.*`), but has **ZERO
codec-QUANTIZER tensors** (grep `quantizer|rvq|vq|vector_quant` = 0). The quantizer (audio → the discrete
16-codebook tokens) is the exact piece needed to make Talker training LABELS from target audio — and it was
not released. (The audio_tower is comprehension features in the wrong space; it cannot substitute.)

### Route A (PREFERRED) — native codec via the Qwen3-TTS tokenizer
Qwen3-TTS DID release its **12 Hz codec tokenizer**. Since the vocoder is shared, the codec is very likely
shared too ⇒ use the TTS tokenizer as the missing quantizer: encode character audio → codec tokens (in
Qwen's own space) → train Omni Talker+MTP → decode with our 5a Code2Wav. Keeps everything native, reuses
5a, no foreign vocoder. **GATE: verify the TTS-12Hz codebook == Omni's `code2wav.code_embedding` space
([32768,1024] = 16×2048). One concrete check, not yet run.**

### Route B (FALLBACK) — Mimi codec bolt-on
**Mimi** = Kyutai/Moshi's open streaming neural codec, 12 Hz, ships **both encoder + decoder** (the analysis
side Qwen omitted). Its 12 Hz rate matches Qwen's, which is why community code reaches for it.
**CATCH:** codec tokens are only meaningful to the decoder they were trained with — **Mimi codes ≠ Qwen
codes**. You cannot Mimi-encode-for-labels and keep Qwen's Code2Wav (it would decode Mimi codes to garbage).
To use Mimi you must use it on BOTH ends → **replace Qwen's Code2Wav with Mimi's decoder entirely**. Costs:
(1) discards the 5a Code2Wav work; (2) a new vocoder (Mimi decoder) to run/port in the runtime; (3) voice
character becomes Mimi's reconstruction, not Qwen-native; (4) unverified. Use ONLY if Route A's codec-identity
check fails.

| | Route A (TTS tokenizer) | Route B (Mimi) |
|---|---|---|
| Supplies the missing encoder | ✅ if codec matches | ✅ always |
| Stays in Qwen's native codec space | ✅ | ❌ foreign |
| Reuses 5a Code2Wav | ✅ | ❌ discarded |
| Extra vocoder to port/run | ❌ none | ✅ Mimi decoder |
| Native Qwen voice quality | ✅ | ❌ Mimi's |
| Risk | codec-identity unverified | distribution-drift vs frozen decoder |

### State of the art (checked Hert4/Qwen-Omni-Training-Talker repo)
**No one has publicly demonstrated a working new voice in the Omni Talker.** That repo is experimental
scaffolding ("test" in name, 8 commits, NO audio samples / metrics / success claim). It CORROBORATES our
findings: it "uses Mimi as a proxy codec because the official qwen-tts-tokenizer encoder is not public"
(= missing-quantizer, confirmed by a 2nd party), and its own README admits the Route-B risk: "if the codec
token distribution drifts far from pretraining, audio quality may suffer, since the vocoder is frozen"
(hedged with LR 2e-5). It fine-tunes Talker-only (Thinker frozen) or LoRA r=8 on Thinker+Talker+MTP;
vocoder frozen; needs an 80 GB H100 for the 30B LoRA run. ⇒ A real character voice in Omni is genuinely
novel territory; Route A is the sounder bet because it avoids the foreign-codec drift the repo itself warns about.

### Route C (FUTURE, contingent) — post-hoc voice conversion (voice changer / RVC)
Zero's idea (2026-10-27): if Omni speaks in a built-in voice (ethan/chelsie), add a **voice-conversion (VC)**
model AFTER it — RVC / so-vits-svc style — to recolor the output timbre into the character's voice. This is a
**voice changer**: Omni audio out → VC → character-timbre audio. **Completely decoupled from the Talker/codec**,
so it SIDESTEPS the missing-quantizer blocker entirely (no Talker training, no codec at all). Ties into
[[project_voiceforge]] / Kokoro character-voice work. Cost: a separate VC model to train (on character audio)
+ run in the pipeline; quality = VC fidelity. Deferred — "we'll get there if we get there, assuming the
experiment works." Good as a fallback voice path that needs none of Route A/B's codec machinery.

## PLAN LOCKED 2026-10-27 (Zero's directive — full remaining arc)
1. **Finish base Qwen3-Omni-30B voice** — 5b-3 (talker codes match gold ref) → 5b-4 (orchestrator: text→WAV). Document + prep testing.
2. **Train Tifa CHARACTER into the 30B Thinker** — behavioral LoRA distillation (NOT weight-merge; "like the Omni-7B char"), using `Z:\VC Projects\Tifa-Distill` (`distill_tifa.py` + `tifa_distill.jsonl` + LLaMA-Factory lf_config/lf_data + Goliath launchers). VOICE deferred (blocked on codec encoder) → use a built-in speaker for vocalization testing.
3. **Quantize for a 16 GB GPU** — Thinker imatrix IQ3 (fits 16 GB), voice stack F16/Q8.
4. **Q3 stress test** — does the character+model survive Q3 or get destroyed? This is the experiment. (Tifa-7B char + Q5 worked; Q3 on a distilled 30B-A3B is the open question.)
Note: Tifa character = Qwen2.5-Omni-7B, CANNOT weight-merge into Qwen3-Omni-30B (different family) — the character is TRAINED in, not merged.
