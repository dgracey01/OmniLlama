# OmniLlama — Phase 0: Research (Annotated Bibliography)

Status: ✅ complete (2026-10-16). PDFs of the core papers are in `../papers/`. This gathers every source relevant to executing Qwen3-Omni's native voice path inside llama.cpp/ggml.

---

## A. Primary architecture — the target model
- **Qwen3-Omni Technical Report** — arXiv **2509.17765** (`papers/arxiv-2509.17765.pdf`). The spec. Thinker+Talker+MTP+Code2Wav. **Table 1 component sizes: Talker = 3B-A0.3B MoE (one codec frame/step); MTP ≈ 80M (ultra-lightweight fixed-step AR dense transformer, fills residual RVQ codebooks); Code2Wav ≈ 200M CAUSAL ConvNet (explicitly replaces block-wise DiT/diffusion, streams left-context-only, frame-by-frame).** RVQ tokens @ **12.5 Hz** (1 token = 80 ms audio). Talker conditions on Thinker hidden states / multimodal features + streamed text. NOT stated in text: codebook count, exact layer counts/hidden sizes, output sample rate, which Thinker layer feeds the Talker, ConvNet upsampling factors → **must read released code/config** (Phase 2 gate).
- **Qwen3-Omni-30B-A3B-Instruct** — HF model card + `config.json` (downloaded at `Z:\Qwen3-Omni-30B-A3B-Instruct`). Source of the actual tensor names/shapes for the converter. Apache-2.0.
- **ggml-org Qwen3-Omni GGUF** — shows the EXISTING (mute) conversion maps to arch `qwen3vlmoe`; confirms image+audio INPUT work, no voice OUT. Our converter extends this, not replaces it.

## B. The precedent — voice-out already in llama.cpp (for another model)
- **`tc-mb/llama.cpp-omni`** (GitHub) — working Talker+vocoder voice-OUT in a llama.cpp fork, for **MiniCPM-o**. Structure to mirror: modular GGUFs (LLM / audio / `tts` + `tts-projector` / `token2wav` bundle / vision); TTS = LLaMA-arch model projecting LLM hidden states → audio tokens, interleaved text+speech; **vocoder (HiFiGAN) runs on CPU**, Token2Mel Metal-accelerated → **CPU-vocoding is the proven path**. Streaming via async queues (`omni_init` / `stream_prefill` / `stream_decode`), sliding window (28-token input / 25 stride). Build targets `llama-omni-cli`, `llama-omni-server`. KEY DIFFERENCE: MiniCPM uses a **flow-matching** vocoder; Qwen3-Omni uses a **causal ConvNet** → Qwen is *simpler* to port. Cannot reuse code (MiniCPM ≠ Qwen Talker/codec), can reuse approach.
- **MiniCPM-o 4.5 Technical Report** — arXiv **2604.27393** (`papers/arxiv-2604.27393.pdf`). The architecture `llama.cpp-omni` implements; useful for how a hidden-state→speech-decoder→vocoder chain is structured and streamed end-to-end.

## C. Reference implementation of the Qwen3-Omni voice pipeline (not llama.cpp)
- **vLLM-Omni** — serves Qwen3-Omni's exact Thinker→Talker→Code2Wav pipeline. Docs + Red Hat "Inside the vLLM-Omni architecture" + vLLM blog (2026-07-01). Confirms the three-stage split, Talker loads only when audio requested, per-stage memory reservations, async chunking for low first-audio latency. **Our numerical correctness reference** — diff our ggml outputs against vLLM-Omni's per stage.

## D. Talker voice TRAINING (sibling track — custom voice)
- **`Hert4/Qwen-Omni-Training-Talker`** (GitHub) — experimental LoRA training of the Qwen3-Omni Talker: Mimi codec + 16-codebook MTP, trains thinker+talker+MTP, 24 kHz mono audio. Confirms **16 codebooks** and the Mimi codec; data format fits the Britt-Baron set. For baking a custom voice once the runtime exists.
- **unsloth #3636** — feature request for optimized Talker/MTP voice-clone fine-tuning. Not shipped; experimental.

## E. The codec
- **Moshi / Mimi** — arXiv **2410.00037** (`papers/arxiv-2410.00037.pdf`), Kyutai. Mimi = the open RVQ neural audio codec (12.5 Hz, semantic+acoustic codebooks) Qwen3-Omni's RVQ tokens align with; encoder is in `transformers` (`kyutai/mimi`). Needed to understand the codec-token space Code2Wav decodes and to build training pairs for the voice-training track.

## F. Related LLM-to-speech architectures (design patterns)
- **LLaMA-Omni** — arXiv **2409.06666** (`papers/arxiv-2409.06666.pdf`), ictnlp. LLM + streaming speech decoder driven by hidden states; a clean reference for the hidden-state-tap → non-AR speech decoder pattern and low-latency streaming.

## G. llama.cpp / ggml mechanics (the build substrate)
- **llama.cpp multimodal docs** (`docs/multimodal.md`) — the `mmproj`/mtmd pattern for bolting a non-LLM module into the GGUF ecosystem; the model our converter extends.
- **Omni design discussion** — ggml-org/llama.cpp Discussion **#16552** ("Discuss how to design omni model"): a maintainer notes the backbone runs via libllama but needs a **special output head**, and the streaming speech decoder is a different path. The canonical mainline thinking on omni.
- **Support request** — issue **#17634** ("support Qwen3-omni, Qwen3-TTS, MiniCPM-o") → dup of **#21956**. Mainline is aware; Omni voice "not planned" → we fork.
- **ggml conv ops** — `ggml_conv_1d`, `ggml_conv_1d_ph`, `ggml_conv_transpose_1d`, depthwise `ggml_conv_1d_dw` exist; signatures take stride/pad/dilation, **no causal flag** (causal = one-sided padding). PR **#28738** fixed a batched-`conv_1d` layout bug (gotcha for batched audio). A recent release (Muse Spark, b11671) ships an audio model as "a causal convolutional pyramid with no attention" → **ggml already runs causal-conv audio decoders in-tree** (strong feasibility signal). RFC **#28541** (diffusion audio / LTX-2) tracks conv fixes.
- **llama-tts / Qwen3-TTS in mainline** — reportedly PR **#26254** (verify): standalone Qwen3-TTS cloning in GGUF. The modular-voice fallback, and a reference for codec/vocoder handling already in-tree.

## H. MoE & performance (because we run TWO MoE models — see Phase 1 §MoE)
- **MoE host-memory expert cache** — llama.cpp **v0.7.0** `--moe-cache-mib` (PR #29887): GPU LRU cache of host-resident experts, up to 2.2× decode — but only with a HIGH hit rate (needs a big cache vs total experts).
- **Independent study** — `AbhayXplor/moe-expert-cache-study`: on a 256-expert 35B at small cache, decode went **−64% (slower)**; gains need a pruned/low-expert model or a large cache. ⇒ on a 16 GB card the cache can BACKFIRE; prefer fully-resident.
- **Vulkan MoE kernels** — PR **#17711** (NVIDIA-focused) optimizes Vulkan MoE matmul; Polaris/gfx803 Vulkan MoE perf is immature. **b11647** = AMD Vulkan strided-KV fix (helps Polaris/RDNA3, off on RDNA4).
- **FreeToken** (arXiv 2608.16157) + llama.cpp `--n-cpu-moe` — bandwidth-adaptive / CPU-offloaded MoE serving background.

---

### What's pinned vs. what needs the released code (Phase 2 gate)
PINNED: component types + sizes (Talker 3B-A0.3B MoE, MTP ~80M, Code2Wav ~200M causal ConvNet), 12.5 Hz / 80 ms frames, **16 codebooks** (Hert4), RVQ/Mimi codec, CPU-vocoding viable, ggml has the conv ops + in-tree causal-conv audio precedent, both Thinker & Talker are MoE.
NEEDS CODE/CONFIG: exact codebook count confirmation, Talker/MTP layer counts & hidden sizes, **which Thinker layer feeds the Talker**, Code2Wav layer stack + upsampling factors, output sample rate. These block the converter's exact tensor map — resolve first in Phase 2.
