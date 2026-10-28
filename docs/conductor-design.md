# OmniLlama Conductor — design guideline

*Written for: engineers building the OmniLlama unified runtime.*

The **Conductor** is the single process that turns the unified `qwen3omni` GGUF into a working
**see / hear / think / speak** model. It loads every sub-model from one file and runs the pipeline:

```
input(text / image / audio) → encoders → thinker → text → [optional] voice → audio
```

Today the hard research is done: the voice path (Talker → MTP → Code2Wav) executes in the fork and is
ear-verified, every component converts to GGUF, and the unified single-file is packaged. What is missing
is this conductor — the loader + orchestration that make the pieces act as one model. This document is the
spec for building it.

---

## 1. The unified file it consumes

`general.architecture = qwen3omni`. One file, namespaced so nothing collides:

| Section (tensor prefix) | Component | Arch / KV namespace |
|---|---|---|
| `thinker.*` | brain | `qwen3vlmoe` MoE; **the Qwen tokenizer lives here** (`tokenizer.ggml.*`) |
| `talker.*` | voice autoregressive model | `qwen3omnitalker` MoE |
| `v.*`, `mm.*` | vision ViT (+ projector) | `clip.*` |
| `a.*` (non-`gen`) | audio-understanding encoder | `clip.*` |
| `a.gen.*` | MTP code-predictor + Code2Wav vocoder | `clip.*` (gen-audio) |
| `text_projection.*`, `codec_special_embd` | talker text front-end | — |

The Conductor **must also accept the components as separate files** (the same tensors, un-prefixed), since
that is how they are built and how `omnillama-tts` already consumes the voice half. Treat "unified file"
and "component set" as two front-ends to the same loader.

## 2. Load phase — the core new work

The obstacle: two llama *models* (thinker + talker) plus clip contexts live in one container, but
`llama_model_load` reads exactly one architecture. The Conductor needs a loader that can **instantiate a
sub-model from a tensor-name prefix + an arch-namespaced hparam set**:

```
thinker = load_submodel(gguf, tensor_prefix="thinker.", arch="qwen3vlmoe")
talker  = load_submodel(gguf, tensor_prefix="talker.",  arch="qwen3omnitalker")
```

Two viable implementations:
- **(a)** extend the fork's `llama_model_load` to take an optional tensor-name prefix + arch override, reusing the existing per-arch graph builders; or
- **(b)** a thin custom loader that maps the prefixed tensors into `llama_model` structs and calls the existing graph builders.

Clip/mtmd contexts (input `v.*`/`a.*` and gen-audio `a.gen.*`) load from the same file; `mtmd_init` already
handles a gen-audio-only mmproj (the two `mtmd.cpp` fork fixes), so extend it to read the unified file's
clip tensors and stand up `ctx_v` + `ctx_a` + `ctx_gen_a` together.

**Fallback (do this first):** given separate component files, load each the normal way (thinker via llama,
mmprojs via mtmd, talker via llama, front-end tensors via the gguf API). This is essentially what
`omnillama-tts` already does for the voice half — the Conductor generalizes it, and it lets stages 1–5
below be built and gated *before* the single-file loader exists.

## 3. Runtime pipeline

**Input assembly (chatml):**
- text → Qwen tokenizer → token ids;
- image → vision encoder (`ctx_v`) → image embeds, spliced at the `<|vision_*|>` placeholders (Qwen3-VL convention);
- audio → audio encoder (`ctx_a`) → audio embeds, spliced at the `<|audio_*|>` placeholders.

**Thinker forward:** decode the multimodal sequence and generate the assistant text. If speech is wanted,
also retain the thinker hidden states at `accept_hidden_layer` (see §4).

**Branch:** text-only reply → return text. Speech reply → §4.

## 4. Speech synthesis (already proven — reuse it)

Reuse the verified `omnillama-tts` machinery, now driven by the thinker's *generated* text instead of a
fixed sentence:
- build the talker prompt: `text_projection(thinker.embed_tokens(ids))` + the Omni `assistant_parts`
  layout (tts_bos/pad/eos + codec specials + the chosen speaker row from `codec_special_embd`);
- per frame: Talker decode → greedy `cb0` + last hidden → `GEN_CODE` (MTP → cb1–15 + feedback =
  `cb0` embed + Σ of the 15 residual-codebook embeds) → next talker input = feedback + text overlay
  (`trailing_text_hidden[frame]`, then `tts_pad`) → stop on `codec_eos`;
- windowed `GEN_WAV` (Code2Wav, 24 kHz) → PCM.

**Multimodal-speech nuance:** pure text-out conditions *only* on `text_projection(embed(text))` (verified).
To speak *about* an image/audio, the real Omni also feeds thinker hidden states (`accept_hidden_layer=24`
+ `hidden_projection`) into the talker's text stream — wire this when extending past TTS. A first Conductor
can ship with text-driven speech (speak the generated reply), which matches what is already verified.

## 5. Placement & memory (16 GB AMD target)

- **Thinker (quant) on GPU** (Vulkan/ROCm): IQ3 ≈ 13.5 GB fits 16 GB with KV headroom; Q4/Q5 for 24 GB.
- **Voice stack on CPU**: Code2Wav is CPU-designed (deliberately dodges GPU/Vulkan kernels); the Talker MoE
  (~3 GB Q8) runs on CPU with a small active path, or on GPU if room remains.
- **Sequential phasing**: think, *then* speak — the two phases need not be co-resident at peak, so the
  Conductor can keep the thinker GPU-resident and run the voice on CPU to fit 16 GB.

## 6. API surface

- **CLI:** `omnillama <unified.gguf | component-dir> [--image X] [--audio Y] --prompt "..." [--speak] [--out out.wav]`.
- **Optional server:** OpenAI-style `/v1/chat/completions` (text, multimodal input) + `/v1/audio/speech` (TTS).

## 7. Verification plan — gate each stage against transformers

1. Thinker text decode — match transformers tokens/logits on a text prompt.
2. Vision/audio input — match the thinker's reply to transformers given the same image/clip.
3. Speech — already gated (cb0 trajectory + WAV vs the 5b reference).
4. End-to-end — image + prompt → spoken reply, A/B against the transformers Omni.

## 8. Build order (incremental, each independently runnable)

1. **Sub-model loader** → thinker text generation works. *Gate: text chat.*
2. **In-process voice** → wire thinker text into the `omnillama-tts` path. *Gate: text → spoken reply.*
3. **Vision input.** *Gate: image + prompt → text reply.*
4. **Audio input.** *Gate: audio + prompt → text reply* (highest-unknown stage — audio-understanding at runtime).
5. **Multimodal speech** (hidden-state overlay). *Gate: "describe this image," spoken.*
6. **AMD / 16 GB** — Vulkan/ROCm thinker + CPU voice; verify fit + output.
7. **Server + streaming** (optional).

## Omni-2 vs Omni-3 — what the Conductor must branch on

The Conductor must detect the model **generation** (from config/arch) and dispatch accordingly. These are
not minor config deltas — the **voice stage differs fundamentally**.

| Aspect | Qwen2.5-Omni (omni-2) | Qwen3-Omni (omni-3) | Impact |
|---|---|---|---|
| Thinker | Qwen2.5 **dense** (7B) | Qwen3 **MoE** (30B-A3B, `qwen3vlmoe`) | different arch loader |
| Talker | **DiT audio decoder** driven from thinker hidden states | autoregressive: 1 codec frame/step + separate **MTP** for residual codebooks; MoE | different talker arch + codes mechanism |
| Vocoder | **block-wise DiT (diffusion)** token2wav | **streaming causal ConvNet** Code2Wav, 12.5 Hz | **fundamentally different** — the fork implements ONLY the omni-3 causal-conv path |
| Audio encoder | Whisper-based | AuT encoder | different input mmproj |
| Streaming | needs a context block first (higher TTFA) | incremental per frame | latency/scheduling |

Sources: [Qwen2.5-Omni report](https://arxiv.org/pdf/2503.20215), [Qwen3-Omni report](https://arxiv.org/pdf/2509.17765).
(To verify per model card: the claim that omni-2 is single-codebook vs omni-3 16-codebook — stated by a
third-party repo, not the official reports.)

**What the fork implements today: omni-3 only.** For an omni-2 model the thinker + input encoders could be
made to load (dense Qwen2 arch + Whisper-style encoder, given converters), but its **native DiT voice
decoder is not in the fork** — a separate, larger build (and the approach Qwen abandoned for omni-3).

**But Zero's omni-2 characters never used the native Talker.** Per the AI Library
(`character-locked-omni-thinker.md`), the shipped characters — Tifa, Aerith, Cortana, Deepsex-RP — are
Qwen2.5-Omni-7B **Thinker-only LoRAs**, deployed as a **thinker-only text GGUF + modular Kokoro voice**
(the 2.5 Talker was dropped because it can't learn a custom voice). So for those characters the omni-2
"voice path" is simply **thinker → text → external Kokoro TTS**, which already works and needs no DiT.

**Practical guidance — the voice stage is a dispatch point:**
- `omni3 model` → native causal-conv Code2Wav (built, verified CUDA/Vulkan/ROCm);
- `omni2 character` → thinker text → **Kokoro** (modular, external) — trivial to wire, matches how they ship;
- `omni2 native DiT voice` → **not built** and not needed for Zero's characters (only if raw omni-2 Talker
  speech were ever required).

**Recommended convergence:** re-distill the omni-2 personas onto the **omni-3** thinker so they ride
OmniLlama's verified native voice — giving the characters a *native* Omni voice for the first time, instead
of modular Kokoro. (Deferred: character phase is held.)

## Known mechanisms — do not re-derive

MTP coupling, the talker feedback sum, NEOX 1-D RoPE, the `assistant_parts` layout, the codec special
IDs, and the `text_projection` ResizeMLP are all documented in `docs/phase2c-findings.md` and the 5b
milestones. The reusable C++ entry points are `tools/mtmd/omnillama-tts.cpp` (full text→speech) and the
`GEN_CODE`/`GEN_WAV` processes in the fork's Qwen3-TTS path.
