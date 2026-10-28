#!/usr/bin/env python3
"""OmniLlama Phase 2b (CUT): Omni voice converter — adapts X:/llamacpp/conversion/qwen3tts.py.

NOT standalone. Integration (Phase 2c, on Goliath): copy into `X:/llamacpp/conversion/`, add the
A_GEN_WAV_CODE_EMBD tensor (see code2wav_omni_variant.md), register the classes in
conversion/__init__.py, then run the stock converter against Z:/Qwen3-Omni-30B-A3B-Instruct.

Reuses the Qwen3-TTS llama.cpp voice runtime (Talker backbone=QWEN3TTS→qwen3vl MoE graph,
Code Predictor=A_GEN_CODE_*, Code2Wav=A_GEN_WAV_* in qwen3tts-gen.cpp). This file only emits the
OMNI checkpoint's voice weights into those tensors.

Omni-vs-TTS deltas handled here (all verified against model.safetensors.index.json / 2a dump):
  - Talker:  Omni `talker.*` (MoE) names already match TTS -> inherit Qwen3TTSTalkerModel.
  - CodePred: Omni `talker.code_predictor.*` names match TTS -> inherit the A_GEN_CODE_* mapping.
  - Code2Wav: Omni ships `code2wav.*` in the MAIN safetensors (not TTS's speech_tokenizer/decoder.*),
              with a FLAT `code_embedding [32768,1024]` (16 cb x 2048) instead of RVQ quantizer, and
              NO pre_conv / NO tfm in/out proj. pre_transformer / upsample / DAC map 1:1 otherwise.
              Needs the C++ flat-embed decode variant (code2wav_omni_variant.md).
  - No ECAPA speaker encoder in Omni (that's a TTS cloning feature) -> skip.
  - hidden_projection (multimodal conditioning) deferred to Phase 3 (text-only path uses token-id
    handoff via text_projection, same as TTS).
"""
from __future__ import annotations
from typing import Callable, Iterable, TYPE_CHECKING

import torch

from .base import ModelBase, gguf  # type: ignore
from .qwen3tts import Qwen3TTSTalkerModel, Qwen3TTSSpeakerEncoderModel  # templates

if TYPE_CHECKING:
    from torch import Tensor

OMNI_ARCH = "Qwen3OmniMoeForConditionalGeneration"
T = gguf.MODEL_TENSOR


def _snake_fold(alpha: "Tensor", beta: "Tensor") -> "tuple[Tensor, Tensor]":
    # same fold qwen3tts.py uses so the ggml snake() graph is just mul/sin/sqr/mul/add
    return torch.exp(alpha), 1.0 / (torch.exp(beta) + 1e-9)


# ----------------------------------------------------------------------------- Talker backbone
@ModelBase.register(OMNI_ARCH)
@ModelBase.example("Qwen/Qwen3-Omni-30B-A3B-Instruct")
class Qwen3OmniTalkerModel(Qwen3TTSTalkerModel):
    """Omni Talker -> QWEN3TTS text model. `talker.*` MoE names match the TTS Talker; the QWEN3TTS
    graph reuses the MoE-capable qwen3vl graph. Defer hidden_projection to Phase 3 (text path works
    via text_projection token-id handoff, as in TTS)."""
    @classmethod
    def filter_tensors(cls, item: "tuple[str, Callable[[], Tensor]]"):
        name, _ = item
        if name.startswith("talker.hidden_projection."):
            return None  # Phase 3 (multimodal conditioning)
        return super().filter_tensors(item)


# --------------------------------------------------- Code Predictor (MTP) + Code2Wav (gen side)
@ModelBase.register(OMNI_ARCH)
@ModelBase.example("Qwen/Qwen3-Omni-30B-A3B-Instruct")
class Qwen3OmniVoiceGenModel(Qwen3TTSSpeakerEncoderModel):
    """Code Predictor + Code2Wav. code_predictor.* names match TTS (inherit A_GEN_CODE_* mapping).
    Code2Wav tensors come through the MAIN safetensors here, so handle them in modify_tensors
    (TTS loaded them from a separate speech_tokenizer file via generate_extra_tensors)."""

    @classmethod
    def filter_tensors(cls, item: "tuple[str, Callable[[], Tensor]]"):
        name, gen = item
        if not (name.startswith("talker.code_predictor.")
                or name == "talker.model.codec_embedding.weight"
                or name.startswith("code2wav.")):
            return None
        return ModelBase.filter_tensors((name, gen))  # type: ignore[attr-defined]

    def generate_extra_tensors(self):
        return iter(())  # Omni code2wav is in the main safetensors -> handled in modify_tensors

    # Omni has no RVQ quantizer / pre_conv / tfm in-out proj, so those force-quant rules are N/A;
    # keep the conv/convtranspose dtype rules from the parent (F16/F32) which still apply to our
    # a.gen.wav.* conv tensors.

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> "Iterable[tuple[str, Tensor]]":
        if name.startswith("code2wav."):
            yield from self._map_code2wav(name, data_torch)
            return
        # code_predictor.* and talker.model.codec_embedding handled by the parent (A_GEN_CODE_*)
        yield from super().modify_tensors(data_torch, name, bid)

    def _map_code2wav(self, name: str, w: "Tensor") -> "Iterable[tuple[str, Tensor]]":
        n = name[len("code2wav."):]

        # flat RVQ-baked embedding [32768,1024] = 16 codebooks x 2048 entries (NEW tensor; C++ variant)
        if n == "code_embedding.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_CODE_EMBD), w)
            return

        # ---- pre_transformer (8 layers, native 1024, NO in/out proj) ----
        if n == "pre_transformer.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), w); return
        if n.startswith("pre_transformer.layers."):
            rest = n[len("pre_transformer.layers."):]      # "{bid}.<key>"
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
            rest = n[len("upsample."):]                     # "{bid}.{0|1}.<key>"
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

        # ---- DAC decoder: decoder.0 entry; decoder.{1..4} blocks; decoder.5 post-snake; decoder.6 post-conv
        if n.startswith("decoder."):
            parts = n.split(".")            # e.g. decoder 1 block 2 conv1 conv weight
            didx = int(parts[1])
            if didx == 0 and parts[2] == "conv":
                yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix="." + parts[-1]), w); return
            if didx == 5 and parts[2] in ("alpha", "beta"):
                # fold at conversion; emit both from alpha/beta pair — handle per-tensor (fold needs both)
                self._post_snake = getattr(self, "_post_snake", {}); self._post_snake[parts[2]] = w
                if len(self._post_snake) == 2:
                    a, b = _snake_fold(self._post_snake["alpha"], self._post_snake["beta"])
                    yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".alpha"), a)
                    yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix=".beta"), b)
                return
            if didx == 6 and parts[2] == "conv":
                yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix="." + parts[-1]), w); return
            # blocks 1..4  (bid = didx-1 to match TTS py=bid+1 convention)
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
