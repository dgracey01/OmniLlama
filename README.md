# OmniLlama

**Goal:** make **Qwen3-Omni-30B-A3B speak inside the GGUF / llama.cpp stack** — i.e., add runtime execution of the model's native voice path (**Thinker → Talker → MTP → Code2Wav**) to llama.cpp, so a single lightweight GGUF deployment gives *brain + vision + hearing + its own voice* on Zero's hardware (3090 / 7800 XT / RX 580).

## The problem this solves
- `convert_hf_to_gguf.py` maps Qwen3-Omni to arch `qwen3vlmoe` and **drops the Talker/Code2Wav** → every Omni GGUF is **mute** (image+audio IN, text OUT only). It's a **runtime gap** (llama.cpp has no code to *run* the voice), not a format or quantization limit.
- The native voice only runs in **`transformers` (PyTorch)** — which needs CUDA/ROCm (3090/7800 XT), **not** the llama.cpp stack Zero standardized on, and **not** the RX 580 at all.
- OmniLlama closes that gap: implement the voice execution in llama.cpp/ggml so the voice travels with the GGUF.

## Why it's feasible (scoping headline)
The Qwen3-Omni voice stack beyond the Thinker is **small and ggml-friendly**: Talker **3B-A0.3B** (0.3B active), MTP **~80M**, **Code2Wav ~200M causal ConvNet** (Qwen deliberately replaced diffusion/DiT with a streaming causal ConvNet). ggml already has 1D conv ops and already runs a causal-conv audio decoder in-tree. Talker+MTP are transformers (reuse existing ggml ops); the only genuinely new work is C++ orchestration (two coupled models + per-frame multi-codebook decode) + porting a 200M causal ConvNet (CPU-runnable, dodging GPU/Vulkan kernels). See `docs/phase1-scoping-spike.md`.

## Precedent
**`tc-mb/llama.cpp-omni`** already does Talker+vocoder voice-out in llama.cpp — for **MiniCPM-o** (flow-matching vocoder). Proves the approach; Qwen3-Omni is *simpler* (causal ConvNet, not flow-matching). Can't reuse its code, can reuse its structure.

## Phases
- **Phase 0 — Research** ✅ done → `docs/phase0-research.md` (annotated bibliography; PDFs in `papers/`).
- **Phase 1 — Scoping Spike** ✅ done → `docs/phase1-scoping-spike.md` (architecture, ggml op gap analysis, staged build plan, effort, risks, **GO/NO-GO**).
- **Phase 2 — Converter** (export Talker+MTP+Code2Wav+projector tensors to GGUF). *not started*
- **Phase 3 — Talker+MTP runtime** (coupled graph + per-frame codebook decode loop). *not started*
- **Phase 4 — Code2Wav** (causal ConvNet forward pass, CPU first). *not started*
- **Phase 5 — Streaming + server** (async pipeline, audio-out API). *not started*
- **Phase 6 — Vulkan/580** (optional: move Code2Wav off CPU onto the 580). *not started*

## Hardware
Build/test on **Goliath's 3090** (CUDA, reference numerics). Target deployment: **RX 580** (Vulkan GGUF brain + CPU Code2Wav) and **7800 XT** (ROCm). Source model: `Z:\Qwen3-Omni-30B-A3B-Instruct` (= Goliath E:).

## Status
Phases 0–1 complete (this document set). Verdict: **conditional GO** for a staged build — see the scoping spike's verdict + the open unknowns that require the released Qwen3-Omni code/config before Phase 2.

## Cross-refs
Library: `AI_Library/reference/qwen3-omni.md`, `AI_Library/research/omnillama-scoping.md`. Related: character-locked-omni-thinker (2.5-Omni), voiceforge-pipeline, imatrix-converter.
