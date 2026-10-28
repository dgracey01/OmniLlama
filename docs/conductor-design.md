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

## Supporting both Qwen Omni generations (omni-2 + omni-3)

Goal: one Conductor that loads **either** a Qwen2.5-Omni (omni-2) or a Qwen3-Omni (omni-3) model and runs
it as see/hear/think/speak. The enabling idea: **decouple the brain from the voice** and make every stage
**generation-dispatched from config** — so a model is just `(thinker, input-encoders, voice-backend)`
chosen per generation, not a monolith.

### Core principle: brain ↔ voice is a pluggable pairing
Zero's own setup proves this is the right seam — omni-2 characters already run as `Qwen2.5 thinker +
Kokoro`. So the Conductor treats **voice as an interface**, and any brain pairs with any voice:

```
interface VoiceBackend { speak(text | thinker_hidden, speaker?) -> PCM }
  NativeOmni3Voice   Talker(qwen3omnitalker)+MTP+Code2Wav    [BUILT, verified CUDA/Vulkan/ROCm]
  KokoroVoice        external Kokoro TTS                      [exists — wrap it]
  Qwen3TTSVoice      modular llama-tts (optional)             [exists upstream]
  NativeOmni2Voice   Qwen2.5 DiT token2wav                    [NOT built; optional, not recommended]
```

The thinker always emits text and, when a native voice is selected, exposes the hidden-state tap; the
chosen backend consumes whichever it needs.

### Per-stage dispatch matrix

| Stage | omni-2 (Qwen2.5-Omni-7B) | omni-3 (Qwen3-Omni-30B-A3B) | Built? |
|---|---|---|---|
| detect generation | arch `Qwen2_5Omni…` / dense | arch `Qwen3OmniMoe…` / `qwen3vlmoe` | trivial |
| thinker decode | dense Qwen2 text GGUF | `qwen3vlmoe` MoE GGUF | both load in llama.cpp today |
| vision input | Qwen2.5-VL ViT mmproj | qwen3vl ViT mmproj | omni-3 built; omni-2 convert TBD |
| audio input | Whisper-style encoder | AuT encoder | general runtime gap, both gens |
| **voice** | **Kokoro** (what chars use) · or native DiT | **native Talker+MTP+Code2Wav** | omni-3 built; Kokoro easy; omni-2 DiT not built |
| constants / template | omni-2 ids / specials / speakers | omni-3 ids / specials / speakers | config-read, never hardcode |

### What "handle both" actually requires

**Already built / near-free:** omni-3 full voice; both thinker text decoders (llama.cpp supports dense
Qwen2 and `qwen3vlmoe`); omni-3 input mmproj.

**Bounded new work:**
1. **Generation detector + dispatch** — read arch/config; pick thinker loader, encoders, voice backend.
2. **Voice-backend interface** — formalize the seam above; implement `NativeOmni3Voice` (wrap the
   `omnillama-tts` machinery) and `KokoroVoice` (link/shell Kokoro).
3. **omni-2 thinker + Kokoro end-to-end** — load the 2.5 thinker GGUF, route its text to Kokoro. This
   alone makes **every existing character** run inside the Conductor (the text brain they already have +
   their Kokoro voice). Low effort, high payoff.
4. **Config-driven constants** per generation (stop hardcoding omni-3 IDs in the voice assembly).

**Optional / larger (only if ever needed):**
5. **omni-2 vision/audio input mmproj** — if the omni-2 characters should *see/hear* (they ship text-only
   today). New converters + runtime.
6. **`NativeOmni2Voice` (DiT token2wav)** — person-months of a diffusion vocoder in ggml, the approach
   Qwen abandoned. **Not recommended** — Kokoro covers omni-2 voice, and the convergence path gives native
   voice via omni-3 instead.

### Recommended shape
- **omni-3** → native everything (brain + eyes/ears + native voice). The flagship.
- **omni-2** → thinker + Kokoro (runs the existing characters as-is), optionally + vision/audio input.
- **Don't** build the omni-2 DiT voice; **converge** instead — re-distill omni-2 personas onto the omni-3
  thinker so they inherit native voice. Long-term one brain family (omni-3); omni-2 stays runnable through
  the Kokoro backend for the legacy characters.

Net: the Conductor stays a **thin dispatcher over {brain loader × input-encoder set × pluggable voice
backend}** — which is exactly why both generations fit one runtime without special-casing deep in the
pipeline.

## Known mechanisms — do not re-derive

MTP coupling, the talker feedback sum, NEOX 1-D RoPE, the `assistant_parts` layout, the codec special
IDs, and the `text_projection` ResizeMLP are all documented in `docs/phase2c-findings.md` and the 5b
milestones. The reusable C++ entry points are `tools/mtmd/omnillama-tts.cpp` (full text→speech) and the
`GEN_CODE`/`GEN_WAV` processes in the fork's Qwen3-TTS path.
