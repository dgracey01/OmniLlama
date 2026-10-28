# OmniLlama — Phase 1: Scoping Spike

Status: ✅ complete (2026-10-16). Inputs: Phase 0 research. Output: component/op analysis, MoE+perf plan, staged build, effort, risks, **GO/NO-GO**.

**Question:** can we fork llama.cpp to execute Qwen3-Omni's voice path (Thinker→Talker→MTP→Code2Wav) so a GGUF deployment speaks — and address MoE/perf while we're in there? **Answer: conditional GO.** The architecture is the friendly case; the risks are nameable; one gate (read the released code) precedes the build.

---

## 1. Component map (what we must execute)
| Component | Type | Size | ggml implication |
|---|---|---|---|
| **Thinker** | MoE transformer | 30B-A3B | already runs (arch `qwen3vlmoe`), text+vision+audio IN |
| **Talker** | **MoE** AR transformer | **3B-A0.3B** (0.3B active) | reuses existing ggml transformer+MoE ops; emits codebook-0 @ 12.5 Hz |
| **MTP** | dense AR transformer, fixed-step | **~80M** | existing ops; fills the residual codebooks (16 total per Hert4) |
| **Code2Wav** | **causal ConvNet** (not diffusion) | **~200M** | 1D causal convs + activations; CPU-runnable; the only "new" work |

**Critical:** the voice stack beyond the Thinker is **~3.3B params total, mostly inactive** (Talker 0.3B active + 80M + 200M). It fits *alongside* a resident Thinker with room to spare — this is a fully-resident design, not a spill problem.

## 2. The three build pieces
**P2 — Converter (days).** Extend `convert_hf_to_gguf.py` (in the `conversion/` package) to export Talker + MTP + Code2Wav + the Thinker→Talker projector as GGUF modules (companion files, `mmproj`/mtmd pattern). Talker/MTP are transformers → tensor maps like any LLM; Code2Wav conv tensors need new GGUF naming. **Gated on reading the released config** (§6 unknowns).

**P3 — Talker+MTP runtime (weeks — the real systems work).** C++ orchestration, **not** new math kernels (both are transformers reusing ggml ops incl. MoE matmul):
- tap Thinker hidden states at the right layer → projector,
- run the **Talker as a second coupled context** (own KV cache) stepping with the Thinker,
- **per-frame multi-codebook decode loop**: codebook 0 from Talker → MTP fills codebooks 1..15 (fixed-step).
The novel llama.cpp work = two coupled models + a multi-codebook frame loop (the engine assumes one model / one token stream).

**P4 — Code2Wav (weeks, de-risked).** Port a 200M **causal ConvNet** forward pass: 1D causal convs (= one-sided padding over `ggml_conv_1d`/`_transpose_1d`/`_dw`) + activations, frame-by-frame, left-context only. **Run on CPU first** (tc-mb runs its vocoder on CPU; 200M is cheap) → **dodges GPU/Vulkan kernels entirely.** ggml already runs an in-tree causal-conv audio decoder (Muse Spark), so op coverage is likely sufficient; watch the batched-`conv_1d` layout bug (PR #28738).

**P5 — Streaming + server (weeks).** Async pipeline (Thinker/Talker+MTP/Code2Wav in parallel queues), sliding-window vocoding, audio-out over the server API. Mirror tc-mb's `omni_init`/`prefill`/`decode` shape.

**P6 — Vulkan/580 Code2Wav (optional).** Only if CPU vocoding misses real-time on the 580; move the ConvNet onto Vulkan (greenfield kernels). Likely unnecessary — 200M causal conv on CPU should hit the 12.5 Hz (80 ms/frame) budget.

## 3. ggml op gap analysis
- **Transformers (Thinker/Talker/MTP):** fully covered — attention, MoE matmul, RoPE, norms all exist.
- **Code2Wav convs:** `ggml_conv_1d`, `ggml_conv_1d_ph`, `ggml_conv_transpose_1d`, `ggml_conv_1d_dw` present. **No causal flag → one-sided padding.** In-tree causal-conv audio decoder precedent (Muse Spark) = strong signal the op set suffices. **Risk:** a specific dilation/grouping/weight-norm config Code2Wav uses may need an op addition — resolvable, small. **Gotcha:** batched `conv_1d` layout (PR #28738) — keep batch=1 for streaming or use the fix.
- **Net:** likely **zero or few new kernels** if Code2Wav runs on CPU. This is the biggest positive vs. the original "person-months of new kernels" fear.

## 4. MoE & performance (folded in per Zero's ask — "while we're cooking our own recipe")
**We run TWO MoE models coupled (Thinker 30B-A3B + Talker 3B-A0.3B). MoE handling is required twice — this drives fork-base choice and perf.**
- **Fork base:** branch from **v0.7.0+** so we inherit `qwen3vlmoe` (Thinker), the `--moe-cache-mib` expert cache, and the **b11647 AMD Vulkan strided-KV fix** (helps Polaris/RDNA3). Verify the base runs a *second* MoE context cleanly (the Talker) — llama.cpp wasn't designed for two MoE models in one process; this is a **compatibility item to test early in P3**.
- **580 (16 GB) strategy — fully resident, NOT expert-cache:** the independent study shows `--moe-cache-mib` can go **−64% (slower)** on a 16 GB / many-expert model (low hit rate). So: **imatrix IQ3 Thinker fully resident** (~13 GB) + Talker(0.3B active)/MTP(80M)/Code2Wav(200M) fit alongside → no expert spill, no cache gamble. q8_0 KV cache, ≤16k ctx. This is why the small voice stack matters: it fits *with* the brain.
- **Vulkan MoE kernel perf:** PR #17711 (NVIDIA-focused) optimizes Vulkan MoE matmul; **Polaris Vulkan MoE is the real perf ceiling** (~10–20 tok/s est.). Fold-in opportunities: pull current Vulkan MoE kernels, keep b11647, consider depthwise/packing tweaks for gfx803 — but accept Polaris is bandwidth/compute-capped regardless.
- **Real-time budget:** 12.5 Hz = **80 ms per audio frame**. The full Thinker→Talker→MTP→Code2Wav must produce 80 ms of audio in <80 ms for real-time. On the 580 this is the **key perf validation** (if it misses, async/streaming buffering hides it for non-interactive use, or P6 Vulkan vocoder). On the 3090 it's comfortable.
- **Precision:** quantize the **Thinker** (imatrix IQ3 / native 4-bit); keep **Talker/MTP/Code2Wav at Q8_0/fp16** (vocoders break at low bit — they're small, so it's cheap).

## 5. Effort & staging
Revised down from the initial "person-months of kernels" because Code2Wav is a causal ConvNet (not diffusion) and CPU-runnable:
- P2 converter: ~days. P3 Talker+MTP runtime: ~2–4 wk (the real work). P4 Code2Wav CPU: ~2–3 wk. P5 streaming/server: ~2–3 wk. → **~1.5–3 months** for a dev fluent in llama.cpp/ggml; less reusing tc-mb's structure. P6 Vulkan vocoder optional (+weeks).
- **For Zero (directs AI, doesn't write ggml C++):** feasibility = "can a coding agent build it." Deep systems C++ + hardware-in-the-loop debug against a moving codebase → **staged, de-risk-first attempt with real but non-fatal failure risk.** A reasonable bet given the friendly architecture, not a one-shot.

## 6. Risk register
| Risk | Severity | Mitigation |
|---|---|---|
| Exact Code2Wav/MTP/Talker hyperparams, codebook count, Thinker-tap layer, sample rate not in paper | **HIGH (gates P2)** | read released Qwen3-Omni code/config + `kyutai/mimi`; confirm before converter |
| Two coupled MoE contexts unsupported in llama.cpp | MED | prototype the second-MoE-context load early in P3; patch if needed |
| A Code2Wav conv config needs a new ggml op | MED | one-sided-pad existing ops; add op if required (small); CPU fallback |
| Real-time miss on Polaris (80 ms/frame) | MED | async buffering for non-interactive; P6 Vulkan vocoder; or 7800XT/3090 host |
| Numerical drift vs. reference → garbled/robotic audio | MED | diff each stage vs. vLLM-Omni / transformers reference; keep voice stack high-precision |
| Perpetual fork (mainline "not planned") | LOW-MED | pin to a chosen base (Zero already pins versions); document rebase points |

## 7. Deployment targets
- **3090 (CUDA):** build/test + reference numerics; comfortable real-time.
- **7800 XT (ROCm) / 580 (Vulkan):** target. 580 = IQ3 Thinker (Vulkan, resident) + Talker/MTP/Code2Wav (Code2Wav on CPU). Validate the 80 ms/frame budget here.

## 8. VERDICT — conditional GO
**Build it, staged, after one gate.** The architecture is the friendly case (small MoE Talker + ~80M MTP + 200M *causal-ConvNet* vocoder; ggml has the ops + in-tree causal-conv precedent; CPU-vocoding dodges GPU kernels), and MoE/perf concerns resolve to a **fully-resident IQ3 design** that sidesteps the expert-cache trap.
**GATE before Phase 2:** obtain the released Qwen3-Omni **code/config** and resolve the §6-HIGH unknowns (codebook count, layer sizes, Thinker-tap layer, Code2Wav hyperparams, sample rate). Also prototype **loading a second MoE context** (the Talker) on the chosen v0.7.0+ base — the one MoE-compatibility item that could surprise us.
**First de-risk milestone (recommended Phase 2a):** converter exports the modules + a tiny harness proves (1) the hidden-state tap matches transformers, and (2) a CPU Code2Wav forward pass reproduces a reference waveform from reference codec tokens. If both pass, the rest is engineering.

---

## 9. Addendum (2026-10-16): Thinker→Talker handoff RESOLVED (one HIGH unknown closed)
Verified against transformers config + vLLM-Omni + sglang-omni PR #2175:
- **The tap layer is `accept_hidden_layer`** in `Qwen3OmniMoeTalkerConfig`. **Shipped Instruct checkpoint = 24** (transformers default = 18 — checkpoint overrides). vLLM captures `capture_layer_indices=[0, 24]`.
- **Per-generated-token handoff is by `token_id`, NOT hidden states.** sglang PR #2175 ("Talker handoff by token id, delete the unread Thinker hidden capture") rebuilds each generated assistant token from the sampled `token_id` via the Talker's own text embedding + `text_projection`; the per-step mid-layer hidden capture was dead code and was removed (perf win). (Contrast: MiniCPM-o's path still reads hidden states.)
- **BUT prompt-side multimodal conditioning still flows from the Thinker** — vLLM's `thinker2talker` splits hidden states into *prompt* vs *generated* embeddings; only the *generated* side is redundant. The Talker still needs the Thinker's representation of the INPUT so speech reflects audio/visual/emotional context (paper: "conditions on audio and visual multimodal features"). So it is NOT "token IDs only."
- **Impact on P3 (simpler than feared):** no tight per-token mid-layer sync. Decode loop = generated tokens handed off by `token_id` through the Talker's `text_projection`; wire the **prompt conditioning handoff once per turn** (layer-24 hidden states for the prompt span). Perf note (from the PR): do NOT copy hidden states per token.
- **Still-open HIGH unknowns (narrowed):** codebook count (16 per Hert4 — confirm), Talker/MTP layer counts & hidden sizes, Code2Wav layer stack + upsampling factors, output sample rate. Thinker-tap layer is now CLOSED (= 24, prompt-conditioning).

---

## 10. Addendum (2026-10-16): FULL SPEC PINNED from config.json — Phase 2 gate essentially CLEARED
Read directly from `Z:\Qwen3-Omni-30B-A3B-Instruct\config.json` (ground truth) + paper quote verified. Nearly all §6 HIGH unknowns resolved:

| Module | Pinned spec (config.json) |
|---|---|
| **Thinker** | 48 layers, hidden **2048**, MoE **128 experts / 8 per tok**, 32 heads / 4 KV, vocab 152064 |
| **Talker** | **20 layers**, hidden **1024**, MoE **128 experts / 6 per tok**, 16 heads / 2 KV, vocab 3072; `accept_hidden_layer: 24` |
| **Code Predictor (MTP)** | **5 layers** (full_attention), hidden 1024, 16 heads / 8 KV, `num_code_groups: 16`, vocab 2048 |
| **Code2Wav** | **8 layers**, hidden **1024**, decoder_dim 1536, intermediate 3072, RVQ **16 quantizers** (1 semantic), codebook_size 2048, `sliding_window: 72`, **upsample_rates [8,5,4,3] × upsampling_ratios [2,2]** |
| **Audio encoder** | 32 layers, d_model 1280, output_dim 2048, 128 mel bins |
| **Vision encoder** | depth 27, hidden 1152, out_hidden 2048, patch 16 |

**Derived: output = 24 kHz** → 8·5·4·3·2·2 = **1920** upsample × **12.5 Hz** code rate = 24,000 Hz. (Resolves the sample-rate unknown.)

**Codebooks = 16** (`num_code_groups`/`num_quantizers`), 1 semantic + 15 acoustic: Talker emits codebook 0, Code Predictor/MTP fills 1–15. (Confirms Hert4's "16-codebook MTP.")

**Thinker→Talker interface (paper-confirmed design intent):** the report states the Talker "no longer consumes the Thinker's high-level text representations and conditions only on audio and visual multimodal features" — rationale: "for textual content, discrete tokens and embeddings are effectively information-equivalent," and "this decoupling allows external modules to supply text to the Talker" with "distinct system prompts." ⇒ **Text handoff = token IDs only; multimodal INPUT additionally pipes audio/vision ENCODER features (not text hidden states) to the Talker.** The scary per-token hidden-state coupling is gone BY DESIGN → clean two-stage pipeline.

**NEW nuance to confirm in modeling code (only open Code2Wav item):** `code2wav_config` carries `num_attention_heads: 16` + `sliding_window: 72` — so Code2Wav may NOT be pure ConvNet; it may have **local sliding-window(72) attention**. If so, it's a small extra (ggml has sliding-window attn), but the port is "causal conv + local attn," not pure conv. Confirm before P4.

**Gate status:** HIGH unknowns from §6 now essentially CLOSED (sizes, codebooks, sample rate, tap layer, interface). **Remaining before build:** (1) confirm the Code2Wav attention nuance in modeling code; (2) prototype loading a 2nd MoE context (Talker) on a v0.7.0+ base. Verdict upgrades from CONDITIONAL GO → **GO once those two checks pass.**

---

## 11. Addendum (2026-10-16): BOTH gate checks CLOSED from the real tensors → **GO**
Verified against `Z:\...\model.safetensors.index.json` (28,010 tensors; voice modules = 8,267) + speculative-decoding precedent.

**Check 1 — Code2Wav composition (from actual tensor names): HYBRID, attention is LIVE (not pure ConvNet).**
```
code2wav.code_embedding                              # codec-code embedding
code2wav.pre_transformer.layers.N.self_attn.{q,k,v,o}_proj + input_layernorm + mlp + *_layer_scale + norm
                                                     # a small TRANSFORMER front-end WITH self-attention (sliding_window=72 local attn → LIVE)
code2wav.decoder.N.conv + .block.N.{conv,conv1,conv2} + .alpha/.beta
                                                     # conv decoder blocks; alpha/beta = SNAKE activation (BigVGAN-style)
code2wav.upsample.N.N.{conv, dwconv, pwconv1, pwconv2, norm, gamma}
                                                     # ConvNeXt-style upsample (depthwise + pointwise conv + layer-scale)
```
⇒ Port needs: self-attention (ggml has it) + conv_1d / depthwise conv (ggml has them) + **Snake activation (one small custom ggml op)** + elementwise layer-scale. Confirmed ~1 new small op, rest composes from existing. Config's `num_attention_heads:16`/`sliding_window:72` are REAL.

**Check 2 — second MoE context: GREEN.** llama.cpp holds two models per process routinely (speculative decoding `-m`+`-md`; `llama-speculative`). The Talker loads as a second `llama_model`/context. Caveat: Vulkan can serialize when two models share ONE device (issue #23126) → matters for the single-GPU 580; mitigate via device/stream placement or run the small voice stack on CPU.

**Full voice-module tensor map (converter input):**
- **Talker** (MoE): `talker.model.layers.N.{self_attn.{q,k,v,o}_proj + q_norm/k_norm, mlp.{gate, experts.N.{gate,up,down}_proj, shared_expert.*, shared_expert_gate}, input_layernorm, post_attention_layernorm}`, `talker.model.{codec_embedding, norm}`, `talker.codec_head`, `talker.hidden_projection.linear_fc1/fc2` (hidden-state/multimodal conditioning), `talker.text_projection.linear_fc1/fc2` (token-ID handoff).
- **Code Predictor / MTP** (dense, 5L, nested under talker): `talker.code_predictor.model.layers.N.{self_attn.*+q_norm/k_norm, mlp.{gate,up,down}_proj, *layernorm}`, `talker.code_predictor.model.{codec_embedding.N, norm}`, `talker.code_predictor.lm_head.N` (per-codebook heads).
- **Code2Wav**: the block above.

**Build location:** Goliath (`X:\llama.cpp` clone, 3090, model local via E:, CUDA toolchain). TheBeast authors; Goliath compiles/tests via the bridge ([[project_goliath_bridge]]). TheBeast lacks cmake + the `gguf` py pkg.

**VERDICT: GO.** Proceed to Phase 2 (converter) — see `phase2-build.md`.
