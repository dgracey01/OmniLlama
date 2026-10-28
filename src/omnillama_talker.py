#!/usr/bin/env python3
"""OmniLlama 5b-1: Qwen3-Omni Talker backbone -> llama.cpp text GGUF (arch `qwen3omnitalker`).

The Omni talker is driven by SUMMED embeddings (a text stream + a codec stream, added per position;
see modeling_qwen3_omni_moe.py generate / _get_talker_assistant_parts), NOT by text token ids. So unlike
Qwen3-TTS there is NO folded-text table and NO logit-pad: the model's vocab is just the codec vocab
(3072). TOKEN_EMBD = talker.model.codec_embedding; OUTPUT = talker.codec_head (tied in the checkpoint).

Backbone = qwen3 attention (q/k norm, GQA kv=2, head_dim 128) + qwen2moe FFN (128 experts / 6 used +
shared_expert + shared_expert_gate). We subclass Qwen3MoeModel, which (via Qwen2MoeModel) already does
per-expert 3D stacking and shared-expert mapping, and (via qwen3) q/k-norm + head_dim. We only:
  - feed it the talker's text_config as hparams (hidden 1024 / 20L / MoE 128-6), not the thinker's;
  - strip the `talker.` prefix and drop text_projection / hidden_projection / code_predictor (the code
    predictor + code2wav live in the 5a voice mmproj; text_projection is applied orchestrator-side);
  - map codec_embedding -> TOKEN_EMBD and codec_head -> OUTPUT;
  - write a codec-only vocab (no text tokenizer; the model is embedding-driven).

Direct-instantiation driver: src/convert_omni_talker.py (bypasses conversion/__init__.py arch dispatch).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

from .base import ModelBase, gguf
from .qwen import Qwen3MoeModel

if TYPE_CHECKING:
    from torch import Tensor


class Qwen3OmniTalkerModel(Qwen3MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3OMNITALKER

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        talker = hparams["talker_config"]
        tcfg = dict(talker["text_config"])  # the backbone config (hidden 1024, 20L, MoE 128/6, kv2, hd128)
        tcfg["architectures"] = ["Qwen3OmniMoeTalkerForConditionalGeneration"]
        self._talker_config = talker
        # skip Qwen3MoeModel's own hparam reload path for backbone fields by feeding tcfg directly
        super().__init__(dir_model, *args, hparams=tcfg, **kwargs)

    # codec-only vocab (3072); the talker is driven by embeddings, so no real text tokenizer is needed
    def set_vocab(self):
        n = int(self.hparams["vocab_size"])
        names = [f"<|codec_{i}|>" for i in range(n)]
        for k, v in self._talker_config.items():
            if k.endswith("_id") and isinstance(v, int) and 0 <= v < n:
                names[v] = f"<|{k[:-len('_id')]}|>"
        self.gguf_writer.add_tokenizer_model("none")
        # vocab type "none" creates n dummy tokens from this KV (llama-vocab.cpp); without it
        # vocab.n_tokens()==0 and token_embd/output/logits all collapse to width 0.
        self.gguf_writer.add_vocab_size(n)
        self.gguf_writer.add_token_list([t.encode("utf-8") for t in names])
        self.gguf_writer.add_token_types([gguf.TokenType.CONTROL] * n)
        self.gguf_writer.add_eos_token_id(int(self._talker_config["codec_eos_token_id"]))
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)

    @classmethod
    def filter_tensors(cls, item: "tuple[str, Callable[[], Tensor]]"):
        name, gen = item
        if not name.startswith("talker.") or name.startswith("talker.code_predictor."):
            return None
        sub = name[len("talker."):]
        if sub.startswith(("text_projection.", "hidden_projection.")):
            return None
        return super().filter_tensors((sub, gen))

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> "Iterable[tuple[str, Tensor]]":
        if name == "model.codec_embedding.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)
            return
        if name == "codec_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return
        yield from super().modify_tensors(data_torch, name, bid)
