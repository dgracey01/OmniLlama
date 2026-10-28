# Code2Wav — Omni flat-embed variant (C++ changes to X:\llamacpp)

The Qwen3-TTS Code2Wav runtime (`tools/mtmd/models/qwen3tts-gen.cpp::clip_graph_qwen3tts_gen::code2wav`)
is reused AS-IS for `pre_transformer`, `upsample`, and the DAC decoder. **Only the front-end differs**
for Omni. Changes below; expect 1–3 compile/debug iterations on Goliath (can't compile-check off-box).

## Omni vs TTS front-end (verified from weights + qwen3tts-gen.cpp read)
| Stage | TTS | Omni |
|---|---|---|
| code → hidden | RVQ: `quant_first_cb`+`out_proj` (sem) + 15× `quant_rest_cb` summed +`out_proj` (acoustic) → **512** | **flat `code_embedding [32768,1024]`**: 16 codes → get_rows at `g*2048+code`, sum → **1024** |
| pre_conv | 512→1024 causal conv k=3 | **none** (already 1024) |
| tfm in/out proj | 1024↔512 (tfm hidden 512) | **none** (tfm native **1024**, n_head 16, d_head 64) |
| pre_transformer / upsample / DAC | — | **identical** (reuse) |

## 1. `gguf-py/gguf/constants.py`
Add enum + name:
```python
A_GEN_WAV_CODE_EMBD = auto()  # Omni flat RVQ-baked code embedding [n_cb*cb_size, dim]
...
MODEL_TENSOR.A_GEN_WAV_CODE_EMBD: "a.gen.wav.code_embd",
```
Register it for the clip gen loader so `c2w.code_embd` resolves (mirror how A_GEN_WAV_PRE_CONV loads).

## 2. hparams (clip/qwen3tts gen hparams struct + GGUF keys)
Add, read from GGUF metadata the converter writes (add matching `add_*` in qwen3tts.py set_gguf_parameters
for the gen model, or hardcode on the mtmd side as the TTS code2wav hparams already are):
```cpp
bool    wav_flat_code_embed = false; // true for Omni
int32_t wav_n_codebooks     = 16;
int32_t wav_codebook_size   = 2048;
bool    wav_has_pre_conv     = true;  // false for Omni
bool    wav_tfm_has_proj     = true;  // false for Omni
// Omni: wav_tfm hidden = 1024, wav_tfm_n_head = 16 (d_head 64), wav_tfm_n_layer = 8, swa window per config
```

## 3. `qwen3tts-gen.cpp` — flat-embed quant_decode
Add alongside `quant_decode`:
```cpp
// Omni: 16 flat codebooks in one [vocab=n_cb*cb_size, dim=1024] table. Per group g, global id = g*cb_size + code.
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::quant_decode_flat(ggml_tensor * inp_codes) const {
    const auto & c2w = model.c2w;
    const int64_t T   = inp_codes->ne[0];
    const int     ncb = hparams.wav_n_codebooks;     // 16
    const int     cbs = hparams.wav_codebook_size;   // 2048
    ggml_tensor * acc = nullptr;
    for (int g = 0; g < ncb; g++) {
        ggml_tensor * ids = ggml_view_1d(ctx0, inp_codes, T, (size_t) g * inp_codes->nb[1]); // [T] I32 codes for cb g
        if (g != 0) ids = ggml_add(ctx0, ids, ggml_scale_bias(ctx0, ids, 0.0f, (float)(g*cbs))); // +g*cbs offset
        // (simpler: precompute offset ids on the converter/host side, or use ggml_add with a const tensor)
        ggml_tensor * emb = ggml_get_rows(ctx0, c2w.code_embd, ids); // [1024, T]
        acc = acc ? ggml_add(ctx0, acc, emb) : emb;
    }
    cb(acc, "wav_quant_hidden", -1);
    return acc; // [1024, T]
}
```
NOTE: the `+g*cbs` offset on an I32 id tensor is cleaner done by adding a constant I32 tensor (build a
host [T] filled with g*cbs, or add a scalar via ggml). Verify ggml int add semantics; fallback = pass
pre-offset ids from the caller.

## 4. `qwen3tts-gen.cpp::decode` — Omni branch
```cpp
ggml_tensor * hidden = hparams.wav_flat_code_embed ? quant_decode_flat(inp_codes)
                                                   : quant_decode(inp_codes);
ggml_tensor * x;
if (hparams.wav_has_pre_conv) {
    x = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));          // [N,512]
    x = causal_conv1d(x, c2w.pre_conv_w, c2w.pre_conv_b, 1, "pre_conv"); // [N,1024]
} else {
    x = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));          // Omni: hidden already [1024,N] -> [N,1024]
}
ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x));   // [1024,N]
if (hparams.wav_tfm_has_proj) { cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c2w.tfm_in_proj_w, cur), c2w.tfm_in_proj_b); }
for (int il = 0; il < hparams.wav_tfm_n_layer; il++) cur = tfm_layer_forward(cur, c2w.tfm_layers[il], il);
cur = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
cur = ggml_mul(ctx0, cur, c2w.tfm_output_norm_w);
if (hparams.wav_tfm_has_proj) { cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c2w.tfm_out_proj_w, cur), c2w.tfm_out_proj_b); }
// ... upsample + DAC unchanged ...
```
Make `tfm_in_proj_*`, `tfm_out_proj_*`, `pre_conv_*` optional in the loader (nullable) when the Omni
tensors are absent.

## 5. Converter side (already in omnillama_voice_convert.py)
Emits `A_GEN_WAV_CODE_EMBD` (the flat table) and skips pre_conv / tfm in-out proj. Also must write the
hparam flags (wav_flat_code_embed=true, counts, wav_tfm hidden=1024/n_head=16) — add the `add_*` calls in
the gen model's set_gguf_parameters, OR hardcode the Omni code2wav hparams on the mtmd side behind a
model-id check, mirroring how TTS code2wav hparams are currently hardcoded there.

## 6. Risk / validation
- Int-offset of code ids in ggml is the fiddliest bit — if `ggml_add` on I32 is awkward, pass pre-offset
  ids from the host (build the [16*T] global-id buffer in C++ before the graph).
- Validate numerically: dump `wav_quant_hidden` + compare to a transformers reference for one frame.
- The Talker (MoE) + Code Predictor need no C++ change beyond confirming the QWEN3TTS hparam path reads
  Omni's MoE talker config (num_experts etc.) — test at load.
