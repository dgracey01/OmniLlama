# OmniLlama — Source Links (companion to docs/phase0-research.md)

Papers downloaded to `../papers/`:
- **Qwen3-Omni Technical Report** — https://arxiv.org/abs/2509.17765 → `papers/arxiv-2509.17765.pdf`
- **MiniCPM-o 4.5 Technical Report** — https://arxiv.org/abs/2604.27393 → `papers/arxiv-2604.27393.pdf`
- **Moshi / Mimi codec** — https://arxiv.org/abs/2410.00037 → `papers/arxiv-2410.00037.pdf`
- **LLaMA-Omni** — https://arxiv.org/abs/2409.06666 → `papers/arxiv-2409.06666.pdf`

Code / repos:
- tc-mb/llama.cpp-omni (the precedent) — https://github.com/tc-mb/llama.cpp-omni
- Hert4/Qwen-Omni-Training-Talker (Talker voice training) — https://github.com/Hert4/Qwen-Omni-Training-Talker
- QwenLM/Qwen3-Omni — https://github.com/QwenLM/Qwen3-Omni
- ggml-org/llama.cpp — https://github.com/ggml-org/llama.cpp
- kyutai/mimi (codec, in transformers) — https://huggingface.co/kyutai/mimi
- Qwen/Qwen3-Omni-30B-A3B-Instruct (model) — https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct (local: Z:\Qwen3-Omni-30B-A3B-Instruct)

llama.cpp issues / PRs / docs:
- Omni design discussion #16552 — https://github.com/ggml-org/llama.cpp/discussions/16552
- Support request #17634 → dup #21956 — https://github.com/ggml-org/llama.cpp/issues/17634
- conv_1d batched layout fix #28738 — https://github.com/ggml-org/llama.cpp/pull/28738
- MoE expert cache #29887 (`--moe-cache-mib`) — https://github.com/ggml-org/llama.cpp/pull/29887
- Vulkan MoE kernels #17711 — https://github.com/ggml-org/llama.cpp/pull/17711
- multimodal docs — https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md

Reference implementation / perf:
- vLLM-Omni architecture (Red Hat) — https://developers.redhat.com/articles/2026/07/01/inside-vllm-omni-architecture-serving-qwen3-omni
- vLLM-Omni Qwen3-Omni optimization blog — https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization
- MoE expert-cache study — https://github.com/AbhayXplor/moe-expert-cache-study
