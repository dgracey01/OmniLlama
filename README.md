# OmniLlama

**Make Qwen3-Omni *speak* inside llama.cpp.** OmniLlama is a fork of llama.cpp/ggml that adds
runtime execution of Qwen3-Omni's native voice path — **Thinker → Talker → MTP → Code2Wav** — so a
plain GGUF deployment can synthesize speech, with no PyTorch, no `transformers`, and no access to the
30 B model shards at run time.

```
llama-omnillama-tts  frontend.gguf  talker.gguf  voice_mmproj.gguf  "any text you type"  out.wav  [ethan|chelsie|aiden]
```

That one command turns arbitrary text into a 24 kHz WAV using only the compiled binary and three GGUF
files. **Status: working and verified** on an RTX 3090 (see [Verification](#verification)).

## Models (download)

Quantized **unified** GGUFs — the whole Omni (brain + vision + audio input + talker + Code2Wav) packed
into **one file per quant**, in the fashion of the original single model:

**→ https://huggingface.co/Zero-000/OmniLlama-Qwen3-Omni-30B-A3B-GGUF**

| File | Size | Fits |
|---|---|---|
| `…-OmniLlama-Q5_K_M.gguf` | ~28 GB | 24 GB VRAM |
| `…-OmniLlama-Q4_K_M.gguf` | ~25 GB | 24 GB VRAM |
| `…-OmniLlama-IQ3_M.gguf`  | ~20 GB | 16 GB VRAM |

These use a custom unified arch (`general.architecture = qwen3omni`); the **unified loader that runs them
is in development** (stock llama.cpp / ollama / LM Studio expect a decoder + separate `mmproj` and can't
load them yet). The component voice path is already verified in this fork; the single-file loader is the
remaining piece. Base model only — no character fine-tuning.

---

## ⚠️ Target model — what this is calibrated to

**OmniLlama is calibrated to one specific checkpoint:**

> ### `Qwen/Qwen3-Omni-30B-A3B-Instruct`
> Hugging Face: <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>
> - **Variant:** `Instruct` (the instruction-tuned release — *not* `Thinking`, *not* `Captioner`, *not* the base).
> - **Size:** 30 B total parameters, A3B Mixture-of-Experts (~3 B active).
> - **Source dtype:** `bf16`, 15 safetensor shards (`model-00001-of-00015.safetensors` … `-00015-`).
> - **Reference stack used for conversion + numeric ground truth:** `transformers == 5.8.0`.

This is not a soft dependency. The GGUFs produced here **bake in constants, token IDs, tensor shapes,
and codec-embedding rows read directly from this checkpoint's `config.json` and weights.** A different
Omni release — a different size, the `Thinking` variant, or a future revision with a changed tokenizer,
vocab size, special-token IDs, or codec layout — will **not** work with GGUFs built for this one. If you
target a different checkpoint you must re-run the converters against it and re-derive the constants (see
[`src/probe_textproj.py`](src/probe_textproj.py) and [`src/probe_chatml.py`](src/probe_chatml.py), which
dump exactly what is calibration-sensitive).

### Calibrated values baked into the build

These were read from the checkpoint and are hard-wired into the converters / the C++ front-end. If any
of them differ in your model, regenerate.

**Chat / TTS special token IDs** (`config.json`)

| name | id | name | id |
|---|---|---|---|
| `im_start` | 151644 | `tts_pad` | 151671 |
| `im_end` (= bos/eos) | 151645 | `tts_bos` | 151672 |
| `system` | 8948 | `tts_eos` | 151673 |
| `user` | 872 | `assistant` | 77091 |

**Talker codec special IDs** (`talker_config`) — baked, in this order, into the front-end GGUF tensor
`codec_special_embd` (rows 0–8):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| name | nothink | think_bos | think_eos | pad | bos | eos | spk:chelsie | spk:ethan | spk:aiden |
| id | 2155 | 2156 | 2157 | 2148 | 2149 | 2150 | 2301 | 2302 | 2303 |

`codec_eos_token_id = 2150` is the generation stop token. Default speaker is **ethan**.

**Tensor shapes**

| tensor | shape | note |
|---|---|---|
| `thinker.model.embed_tokens.weight` | `[152064, 2048]` | text-embedding table (front-end `token_embd`) |
| `talker.text_projection` | ResizeMLP `2048→2048` → **SiLU** → `2048→1024` (both Linear + bias) | text-stream projector |
| talker hidden size | `1024` | |
| talker codec vocab | `3072` | |

**Talker backbone hparams** (arch `qwen3omnitalker`): 20 layers, hidden 1024, MoE **128 experts / 6
used** + shared expert, attention heads 16 / KV heads 2 (head_dim 128), expert FFN 384, shared-expert
FFN 768, RoPE **NEOX 1-D** `freq_base = 1e6`, `norm_topk_prob = true`, RMS-norm eps `1e-6`.

---

## The problem it solves

Stock `convert_hf_to_gguf.py` maps Qwen3-Omni to arch `qwen3vlmoe` and **drops the Talker and
Code2Wav** — so every Omni GGUF is **mute** (image + audio *in*, text *out* only). This is a **runtime
gap**: llama.cpp simply had no code to *run* the voice stack; it is not a format or quantization limit.
The native voice otherwise runs only in `transformers`/PyTorch, which needs CUDA/ROCm and never touches
the llama.cpp stack at all. OmniLlama implements the voice execution so the voice travels with the GGUF.

**Precedent:** [`tc-mb/llama.cpp-omni`](https://github.com/tc-mb/llama.cpp) does Talker + vocoder
voice-out in llama.cpp for *MiniCPM-o* (flow-matching vocoder). It proves the approach. Qwen3-Omni is
structurally simpler (causal-conv / transformer vocoder), so the structure transfers even though the
code does not.

## How the voice path works

Text → speech in Qwen3-Omni conditions **only** on `text_projection(embed_tokens(token_ids))` — the
Thinker's reasoning hidden states are used only for *multimodal* input, not for speaking text. That is
the key simplification OmniLlama exploits: the speaker is a stable, self-contained component downstream
of whatever brain produced the text.

Per output frame the Talker and the MTP (multi-token-predictor / code predictor) are **coupled**: the
Talker emits codebook-0 and a hidden state; the 5-layer MTP expands that into codebooks 1–15; the next
Talker input is the **sum over all 16 codebooks of the previous frame** plus the text-stream overlay
(`trailing_text_hidden[frame]`, then `tts_pad`). The 16×T code grid is streamed through Code2Wav
(8-layer SWA-72 pre-transformer + upsampling DAC-style causal ConvNet) to 24 kHz PCM.

The fork reuses its existing **Qwen3-TTS** `GEN_CODE` / `GEN_WAV` machinery for the Omni per-frame
loop; the only Omni-specific delta is the changing text overlay. Full derivation in
[`docs/phase2c-findings.md`](docs/phase2c-findings.md).

## The three GGUFs

All three are derived from the one target checkpoint by the converters in [`src/`](src/):

| file | arch | ~size | contents | converter |
|---|---|---|---|---|
| **front-end** | `qwen3vlmoe` | 641 MB | Qwen tokenizer + `token_embd` [152064,2048] + `text_projection` + `codec_special_embd` [9,1024] | [`omnillama_textfe.py`](src/omnillama_textfe.py) |
| **talker** | `qwen3omnitalker` | 5.9 GB (F16) | Talker backbone (codec vocab 3072) | [`omnillama_talker.py`](src/omnillama_talker.py) |
| **voice mmproj** | mmproj (gen-audio) | — (289 tensors) | the full MTP code-predictor + Code2Wav vocoder | [`omnillama_voice.py`](src/omnillama_voice.py) |

The front-end GGUF is what makes the C++ tool self-contained: it carries the tokenizer and the two
small text-side weight groups, so no `transformers` / model shards are needed to turn text into the
Talker prompt at run time.

## Build

Built on Goliath (RTX 3090) with VS 2026 Insiders + CMake/Ninja + CUDA 12.9:

```bat
call "...\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER=".../CUDA/v12.9/bin/nvcc.exe" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-omnillama-tts --config Release -j
```

(The CLI tool is `llama-completion`; there is no `llama-cli` in this fork. Arch 86 = the 3090.)

### Converting the GGUFs (one-time, needs the checkpoint + `transformers`)

```bat
python -I src\convert_omni_talker.py     ::  -> talker GGUF
python -I src\convert_omni_voice.py      ::  -> voice mmproj GGUF
python -I src\convert_omni_textfe.py     ::  -> front-end GGUF
```

Each driver points at the checkpoint (`Qwen3-Omni-30B-A3B-Instruct`) and writes under its own out dir.

## Usage

```bat
llama-omnillama-tts ^
  frontend.gguf  talker.gguf  voice_mmproj.gguf ^
  "Hello, this is a test of the Omni voice." ^
  out.wav  ethan
```

Other tools, in build order (each proves one stage against the PyTorch reference):

| tool | stage |
|---|---|
| `llama-talker-test` | 5b-1: Talker GGUF loads + produces well-formed codec logits |
| `llama-talker-embed-test` | 5b-3: Talker backbone faithful to `transformers` (teacher-forced) |
| `llama-talker-tts-test` | 5b-4: end-to-end text→WAV for a fixed reference sentence |
| `llama-mtmd-c2w-test` | 5a: Code2Wav vocoder path |

The Python front-end `src/build_talker_prompt.py` is the Tier-1 equivalent of the self-contained tool —
it builds the Talker prompt for any sentence using `transformers`, for A/B against the C++ path.

## Verification

The C++ self-contained path was checked **bit-faithfully** against the `transformers` reference on the
3090:

- **Tokenization** — C++ produced the same 36 token IDs as `transformers.apply_chat_template` for the
  test sentence.
- **Prompt assembly** — `talker_input = 19`, `trailing = 23`, identical to the Python front-end.
- **Decode** — greedy codebook-0 trajectory `1049, 1102, 1252, 1605, 1860 …` identical to the
  ear-verified Python-front-end run (MTP sampling then diverges, as expected, since it samples).
- **Output** — 63 frames, clean `codec_eos`, 5.04 s, RMS 0.054; `OMNILLAMA_TTS_PASS`.
- Earlier the fixed-sentence (5b-4) and Python-front-end arbitrary-text (5b-5 Tier-1) WAVs were
  confirmed intelligible by ear.

## Fork edits

Two edits to `tools/mtmd/mtmd.cpp` let a **gen-audio-only** mmproj load (the Omni voice stack has no
input/speaker encoder): a load-guard and a null-deref fix. Because the upstream `llama.cpp` working tree
is *not* under version control locally, both are tagged in-source with `// OmniLlama:` and captured as a
re-appliable patch in [`docs/mtmd-fork-fixes.md`](docs/mtmd-fork-fixes.md); the new tool sources,
converters, and CMake targets are listed there too. After any engine pull:
`grep -c "OmniLlama:" tools/mtmd/mtmd.cpp` should be ≥ 2.

## Repository layout

```
src/     converters, C++ tool sources (mirrors of the files that live in the llama.cpp fork), probes
docs/    phase findings + build runbook + the fork-fix patch
papers/  reference PDFs (Qwen3-Omni / vocoder / MTP background)
refs/    extracted shapes + source list
```

## Roadmap

- **Quantization** — Thinker to IQ3 for the 16 GB RX 580 target; keep the voice stack near-lossless
  (Talker ≤ Q8, MTP/Code2Wav mmproj F16 — codebook/vocoder quantization is where audio quality dies);
  then a Q3 stress test.
- **Character distill** — behavioral LoRA into the Thinker (e.g. a named character) so it *speaks in
  character*. The distill must leave `embed_tokens` and `text_projection` frozen, or the front-end GGUF
  must be re-exported.

## Credits / license

Base model © Alibaba / Qwen, **Qwen3-Omni-30B-A3B-Instruct** (checkpoint metadata reports
`license.name = apache-2.0`); consult the model card for terms. Built on `llama.cpp` (MIT). This
repository holds the OmniLlama converters, C++ tool sources, and documentation.
