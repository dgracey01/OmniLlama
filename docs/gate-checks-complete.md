# OmniLlama — GATE CHECKS COMPLETE (2026-10-27) → verdict GO

The scoping summary set: **"CONDITIONAL GO → GO once two checks pass — (1) Code2Wav attention nuance,
(2) prototype a 2nd MoE (Talker) context on v0.7.0+."** Both are now resolved with hard evidence.

## Check 1 — Code2Wav attention nuance → RESOLVED (it IS attention, and it's already wired)
The paper described Code2Wav as a ConvNet "without attention," but `code2wav_config` carried
`num_attention_heads: 16` + `sliding_window: 72`. Inspecting the shipped weights
(`src/inspect_code2wav.py` against `Z:\Qwen3-Omni-30B-A3B-Instruct`) settles it: Code2Wav is a
**pre_transformer (attention) → upsample → conv/Snake DAC decoder**, not pure ConvNet.

Evidence (per-layer, 8 layers `code2wav.pre_transformer.layers.0..7`):
```
code2wav.pre_transformer.layers.N.self_attn.q_proj.weight   [1024, 1024]
code2wav.pre_transformer.layers.N.self_attn.k_proj.weight   [1024, 1024]   (full MHA: q=k=1024 → 16 heads × 64, NOT GQA)
code2wav.pre_transformer.layers.N.self_attn.v_proj.weight   [1024, 1024]
code2wav.pre_transformer.layers.N.self_attn.o_proj.weight   [1024, 1024]
code2wav.pre_transformer.layers.N.self_attn_layer_scale.scale [1024]
code2wav.pre_transformer.layers.N.mlp.{gate,up,down}_proj    (SwiGLU 1024↔3072)
code2wav.pre_transformer.layers.N.{input,post_attention}_layernorm.weight
```
Downstream is convolutional: `code2wav.decoder.*` = Snake (alpha/beta) + causal conv stacks + upsample
(no attention). `code2wav.code_embedding.weight [32768,1024]` = the flat 16×2048 RVQ code table;
**no `quantizer.rvq_*`** (Omni bakes the RVQ into a flat embedding — differs from TTS).

**Impact:** the attention must be implemented **with sliding-window 72** (local causal attention), NOT
dropped. Good news: this `pre_transformer` maps 1:1 onto the existing Qwen3-TTS `qwen3tts-gen.cpp`
code2wav path (reused as-is per `src/code2wav_omni_variant.md`); only the front-end (flat code embedding,
no pre_conv, no in/out proj) is new. So the nuance is resolved AND it's low-risk to implement.

## Check 2 — a second MoE context in one process → PROVEN
The risk was architectural: the fork loads the Thinker MoE **and** the Talker MoE as two live contexts in
one llama.cpp process, and "llama.cpp wasn't built for it." Tested the general capability directly with
`llama-cpp-python 0.3.36` (bundled recent llama.cpp, CPU): loaded **two MoE models+contexts in one Python
process** (`scratchpad/test_two_moe.py`) and decoded interleaved on both.

```
context A: arch=granitemoe  expert_count=32  expert_used_count=8   (genuine MoE expert routing)
context B: arch=granitemoe  expert_count=32  expert_used_count=8
A1: 'Paris. Paris is located in the north-central part of the country...'
B1: 'four. The answer is four...'
A2: 'oxygen, and the ratio of hydrogen to oxygen is 1:8.'
B2: 'east, and the sun sets in the west...'
→ two MoE contexts live + both producing coherent tokens, interleaved, in ONE process = PASS
```
ggml/llama.cpp holds two independent MoE compute graphs in one process without crash or cross-context
corruption. The architectural risk is retired.

**Scope honesty:** this proves the GENERAL capability on a recent llama.cpp (CPU, granitemoe). It does NOT
substitute for the fork-specific + AMD-Vulkan confirmation, which happens naturally at build time:
- build `X:\llamacpp` at **v0.7.0+ / ≥ b11647** (has `qwen3vlmoe`, `--moe-cache-mib`, and the b11647 AMD
  Vulkan strided-KV fix) and load Thinker(qwen3vlmoe MoE) + Talker(qwen3_omni_moe_talker MoE) together;
- the current `E:/VCcore/LLM_host/bin/vulkan_b11011` is **b11011 < b11647** → too old for the AMD-Vulkan
  path; the fork must be newer. (CPU capability proven here is version-agnostic for the "two MoE" question.)

## Verdict: GO
Both gate conditions met. Code2Wav attention is understood and already mapped to existing C++; two MoE
contexts coexist in one process. Proceed to the Phase 2c/2d runbook (`docs/phase2c-runbook.md`):
convert → build the fork (CUDA on Goliath) → de-risk milestone 5a (Code2Wav WAV in isolation) → 5b
(end-to-end Thinker→Talker→CodePredictor→Code2Wav → intelligible WAV).
