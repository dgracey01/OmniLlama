#!/usr/bin/env python3
"""OmniLlama 5a: Qwen3-Omni voice (Code Predictor + Code2Wav) -> llama.cpp mmproj GGUF.

Reuses the Qwen3-TTS voice runtime (A_GEN_CODE_* code predictor + A_GEN_WAV_* code2wav in
tools/mtmd/models/qwen3tts-gen.cpp). This emits the OMNI checkpoint's gen-side weights into
those tensors. The Talker TEXT backbone is a separate, harder conversion (Phase 5b) and is NOT
needed for the 5a Code2Wav-in-isolation test, which drives GEN_WAV from fixed codes.

Omni-vs-TTS deltas handled here (verified against config.json + the 28,010-tensor index):
  - Code2Wav lives in the MAIN safetensors as `code2wav.*` (TTS loads it from speech_tokenizer/).
  - Code2Wav front-end is a FLAT `code_embedding [32768,1024]` (16 cb x 2048), NOT an RVQ quantizer,
    and has NO pre_conv / NO tfm in/out proj (tfm native 1024). -> new tensor A_GEN_WAV_CODE_EMBD;
    the C++ decode takes the flat-embed branch (quant_decode_flat, mean/16). pre_transformer /
    upsample / DAC map 1:1 to A_GEN_WAV_* otherwise.
  - Code Predictor (MTP) `talker.code_predictor.*` matches TTS -> inherit A_GEN_CODE_* mapping;
    Omni has no `small_to_mtp_projection` (A_GEN_CODE_PROJ_IN simply absent).
  - No ECAPA speaker-clone encoder in Omni -> has_audio_encoder=False, only the gen-audio encoder.
  - talker_config nests hidden_size under talker_config["text_config"] (TTS has it at the top).

Direct-instantiation converter (see src/convert_omni_voice.py); it does NOT rely on the
conversion/__init__.py arch dispatch (Omni's arch is already mapped to qwen3vl for the Thinker).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

from .base import ModelBase, MmprojModel, gguf
from .qwen3tts import Qwen3TTSSpeakerEncoderModel

if TYPE_CHECKING:
    from torch import Tensor

T = gguf.MODEL_TENSOR


def _snake_fold(alpha: "Tensor", beta: "Tensor") -> "tuple[Tensor, Tensor]":
    # same fold qwen3tts uses so the ggml snake() graph is just mul/sin/sqr/mul/add
    return torch.exp(alpha), 1.0 / (torch.exp(beta) + 1e-9)


class Qwen3OmniVoiceGenModel(Qwen3TTSSpeakerEncoderModel):
    """Omni Code Predictor + Code2Wav -> QWEN3TTS_GEN mmproj. No speaker encoder."""

    has_vision_encoder = False
    has_audio_encoder = False  # Omni has no ECAPA speaker-clone encoder

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        talker = hparams["talker_config"]
        # talker hidden lives under talker_config["text_config"] in Omni (TTS has it at the top)
        talker_text = talker.get("text_config", talker)
        talker_hidden = talker_text["hidden_size"]
        # load_hparams set text_config = thinker_config.text_config (hidden 2048); override to the
        # talker hidden (1024), which is the gen-audio projection / h_state dim clip.cpp expects
        hparams["text_config"] = {"hidden_size": talker_hidden}
        # MmprojModel.__init__ requires a vision/audio config to exist + a block_count; Omni has no
        # input encoder, so inject a minimal audio_config (never written; set_gguf_parameters is
        # fully overridden below). block_count is cosmetic for the gen tensors (explicit bids).
        cpc = talker["code_predictor_config"]
        hparams["audio_config"] = {"num_hidden_layers": cpc["num_hidden_layers"]}
        # skip the parent (TTS) __init__: it reads talker_config["hidden_size"] and
        # speaker_encoder_config["n_layers"], neither of which exists in Omni
        MmprojModel.__init__(self, dir_model, *args, hparams=hparams, **kwargs)
        self._wav_config_cache = None

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("audio_config")

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        # gen-audio (code predictor + code2wav) only; no speaker/audio INPUT encoder for Omni
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_GEN)

        cpc = self.global_config["talker_config"]["code_predictor_config"]
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)  # talker hidden (1024)
        self.gguf_writer.add_gen_audio_embedding_length(cpc["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(cpc["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(cpc["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(cpc["num_attention_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(cpc.get("num_key_value_heads", cpc["num_attention_heads"]))
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(cpc.get("rms_norm_eps", 1e-5))
        # code2wav hparams (tfm 1024/8L/swa72, flat-embed) are derived on the clip.cpp side from the
        # presence + shape of a.gen.wav.code_embd, so nothing more is written here.

    @classmethod
    def filter_tensors(cls, item: "tuple[str, Callable[[], Tensor]]"):
        name, gen = item
        if not (name.startswith("talker.code_predictor.")
                or name == "talker.model.codec_embedding.weight"
                or name.startswith("code2wav.")):
            return None
        return MmprojModel.filter_tensors((name, gen))

    def generate_extra_tensors(self) -> "Iterable[tuple[str, Tensor]]":
        return iter(())  # Omni code2wav is in the main safetensors -> handled in modify_tensors

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> "Iterable[tuple[str, Tensor]]":
        if name.startswith("code2wav."):
            yield from self._map_code2wav(name, data_torch)
            return
        # code_predictor.* and talker.model.codec_embedding handled by the TTS parent (A_GEN_CODE_*)
        yield from super().modify_tensors(data_torch, name, bid)

    def _map_code2wav(self, name: str, w: "Tensor") -> "Iterable[tuple[str, Tensor]]":
        n = name[len("code2wav."):]

        # flat RVQ-baked embedding [32768,1024] = 16 codebooks x 2048 (NEW tensor; C++ flat variant)
        if n == "code_embedding.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_CODE_EMBD), w)
            return

        # ---- pre_transformer (8 layers, native 1024, NO in/out proj) ----
        if n == "pre_transformer.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), w); return
        if n.startswith("pre_transformer.layers."):
            rest = n[len("pre_transformer.layers."):]      # "{lid}.<key>"
            lid = int(rest.split(".", 1)[0])
            key = rest.split(".", 1)[1]
            tfm = {
                "input_layernorm.weight":          T.A_GEN_WAV_TFM_ATTN_NORM,
                "self_attn.q_proj.weight":         T.A_GEN_WAV_TFM_ATTN_Q,
                "self_attn.k_proj.weight":         T.A_GEN_WAV_TFM_ATTN_K,
                "self_attn.v_proj.weight":         T.A_GEN_WAV_TFM_ATTN_V,
                "self_attn.o_proj.weight":         T.A_GEN_WAV_TFM_ATTN_OUT,
                "self_attn_layer_scale.scale":     T.A_GEN_WAV_TFM_ATTN_SCALE,
                "post_attention_layernorm.weight": T.A_GEN_WAV_TFM_FFN_NORM,
                "mlp.gate_proj.weight":            T.A_GEN_WAV_TFM_FFN_GATE,
                "mlp.up_proj.weight":              T.A_GEN_WAV_TFM_FFN_UP,
                "mlp.down_proj.weight":            T.A_GEN_WAV_TFM_FFN_DOWN,
                "mlp_layer_scale.scale":           T.A_GEN_WAV_TFM_FFN_SCALE,
            }.get(key)
            if tfm is not None:
                yield (self.format_tensor_name(tfm, lid), w)
            return

        # ---- upsample: 2 stages, each .0.conv (ConvTranspose) + .1 ConvNeXt (dw/norm/pw1/pw2/gamma)
        if n.startswith("upsample."):
            rest = n[len("upsample."):]                     # "{uid}.{0|1}.<key>"
            uid = int(rest.split(".", 1)[0])
            sub = rest.split(".", 1)[1]
            up = {
                "0.conv.weight":        (T.A_GEN_WAV_UP_CONV, ".weight"),
                "0.conv.bias":          (T.A_GEN_WAV_UP_CONV, ".bias"),
                "1.dwconv.conv.weight": (T.A_GEN_WAV_UP_DWCONV, ".weight"),
                "1.dwconv.conv.bias":   (T.A_GEN_WAV_UP_DWCONV, ".bias"),
                "1.norm.weight":        (T.A_GEN_WAV_UP_NORM, ".weight"),
                "1.norm.bias":          (T.A_GEN_WAV_UP_NORM, ".bias"),
                "1.pwconv1.weight":     (T.A_GEN_WAV_UP_PW1, ".weight"),
                "1.pwconv1.bias":       (T.A_GEN_WAV_UP_PW1, ".bias"),
                "1.pwconv2.weight":     (T.A_GEN_WAV_UP_PW2, ".weight"),
                "1.pwconv2.bias":       (T.A_GEN_WAV_UP_PW2, ".bias"),
                "1.gamma":              (T.A_GEN_WAV_UP_GAMMA, ""),
            }.get(sub)
            if up is not None:
                tid, suffix = up
                yield (self.format_tensor_name(tid, uid, suffix=suffix), w)
            return

        # ---- DAC: decoder.0 entry; decoder.{1..4} blocks; decoder.5 post-snake; decoder.6 post-conv
        if n.startswith("decoder."):
            parts = n.split(".")            # e.g. decoder 1 block 2 conv1 conv weight
            didx = int(parts[1])
            if didx == 0 and parts[2] == "conv":
                yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix="." + parts[-1]), w); return
            if didx == 5 and parts[2] in ("alpha", "beta"):
                self._post_snake = getattr(self, "_post_snake", {}); self._post_snake[parts[2]] = w
                if len(self._post_snake) == 2:
                    a, b = _snake_fold(self._post_snake["alpha"], self._post_snake["beta"])
                    yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a)
                    yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), b)
                return
            if didx == 6 and parts[2] == "conv":
                yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix="." + parts[-1]), w); return
            # blocks 1..4 -> bid 0..3 (TTS convention: py = bid+1)
            bid = didx - 1
            if parts[2] == "block":
                blk = int(parts[3])
                if blk == 0 and parts[4] in ("alpha", "beta"):
                    buf = getattr(self, "_dac_up_snake", {}); buf.setdefault(bid, {})[parts[4]] = w
                    self._dac_up_snake = buf
                    if len(buf[bid]) == 2:
                        a, b = _snake_fold(buf[bid]["alpha"], buf[bid]["beta"])
                        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".alpha"), a)
                        yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix=".beta"), b)
                    return
                if blk == 1 and parts[4] == "conv":
                    yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix="." + parts[-1]), w); return
                if blk in (2, 3, 4):  # 3 residual units -> xid 0,1,2
                    xid = blk - 2
                    sub = ".".join(parts[4:])  # act1.alpha / conv1.conv.weight / act2.beta / conv2.conv.bias
                    if sub.startswith("act1.") or sub.startswith("act2."):
                        which = "act1" if sub.startswith("act1.") else "act2"
                        comp = sub.split(".", 1)[1]  # alpha|beta
                        buf = getattr(self, "_dac_res_snake", {})
                        buf.setdefault((bid, xid, which), {})[comp] = w
                        self._dac_res_snake = buf
                        if len(buf[(bid, xid, which)]) == 2:
                            a, b = _snake_fold(buf[(bid, xid, which)]["alpha"], buf[(bid, xid, which)]["beta"])
                            tid = T.A_GEN_WAV_DAC_RES_ACT1 if which == "act1" else T.A_GEN_WAV_DAC_RES_ACT2
                            nm = gguf.TENSOR_NAMES[tid].format(bid=bid, xid=xid)
                            yield (nm + ".alpha", a); yield (nm + ".beta", b)
                        return
                    if sub.startswith("conv1.conv.") or sub.startswith("conv2.conv."):
                        tid = T.A_GEN_WAV_DAC_RES_CONV1 if sub.startswith("conv1") else T.A_GEN_WAV_DAC_RES_CONV2
                        nm = gguf.TENSOR_NAMES[tid].format(bid=bid, xid=xid)
                        yield (nm + "." + parts[-1], w)
                        return
            return
        # anything unmatched: drop (surfaces as a missing-tensor error at load if the graph needs it)
        return
