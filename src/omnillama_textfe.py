#!/usr/bin/env python3
r"""OmniLlama 5b-5 Tier-2: Qwen3-Omni TEXT FRONT-END -> compact GGUF (arch `qwen3vlmoe`, vocab-loadable).

The self-contained C++ orchestrator needs, for arbitrary text, exactly what build_talker_prompt.py does
in Python: tokenize chatml -> thinker embed_tokens -> talker text_projection (ResizeMLP 2048->2048->SiLU
->1024) -> Omni special-token layout. This converter bakes those pieces into ONE small GGUF so no
transformers / 30B shards are needed at run time:
  - the Qwen BPE tokenizer (reused verbatim from Qwen3OmniMoeTextModel.set_vocab);
  - token_embd.weight = thinker.model.embed_tokens.weight  [152064, 2048];
  - text_projection.linear_fc1/linear_fc2 .weight/.bias     (talker.text_projection);
  - codec_special_embd.weight = the talker codec_embedding rows for the Omni special ids (so the C++
    never has to read rows out of the talker GGUF at run time).

Arch stays QWEN3VLMOE (in the fork's C++ arch table) so a vocab-only load works; the block tensors are
simply absent (the C++ loads this file vocab-only, and reads the three baked tensors via the gguf API).

Direct-instantiation driver: src/convert_omni_textfe.py.  Run on Goliath via the .venv python.
"""
from __future__ import annotations

import glob
import os
from typing import Callable, Iterable, TYPE_CHECKING

import torch

from .base import ModelBase, gguf
from .qwen3vl import Qwen3OmniMoeTextModel

if TYPE_CHECKING:
    from torch import Tensor

# talker codec_embedding rows baked for the prompt assembly. ORDER IS THE C++ CONTRACT -- do not reorder.
#   0 codec_nothink  1 codec_think_bos  2 codec_think_eos  3 codec_pad  4 codec_bos  5 codec_eos
#   6 speaker chelsie  7 speaker ethan  8 speaker aiden
CODEC_SPECIAL_IDS = [2155, 2156, 2157, 2148, 2149, 2150, 2301, 2302, 2303]


class Qwen3OmniTextFrontEnd(Qwen3OmniMoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3VLMOE  # re-declared: base __init_subclass__ requires it explicitly
    # inherits the Qwen BPE set_vocab (+ im_end bos/eos fix)

    @classmethod
    def filter_tensors(cls, item: "tuple[str, Callable[[], Tensor]]"):
        name, gen = item
        # keep ONLY the thinker token-embedding table; drop the entire 30B backbone + talker + code2wav
        if name != "thinker.model.embed_tokens.weight":
            return None
        return super().filter_tensors((name, gen))  # -> "model.embed_tokens.weight" -> token_embd.weight

    def generate_extra_tensors(self) -> "Iterable[tuple[str, Tensor]]":
        want_tp: dict[str, "Tensor"] = {}
        codec_w: "Tensor | None" = None
        for shard in sorted(glob.glob(os.path.join(self.dir_model, "model-*-of-*.safetensors"))):
            from safetensors import safe_open
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("talker.text_projection."):
                        want_tp[k[len("talker.text_projection."):]] = f.get_tensor(k)
                    elif k == "talker.model.codec_embedding.weight":
                        codec_w = f.get_tensor(k)
        assert want_tp, "talker.text_projection.* not found in shards"
        assert codec_w is not None, "talker.model.codec_embedding.weight not found in shards"
        for sub, t in sorted(want_tp.items()):            # linear_fc1.weight/.bias, linear_fc2.weight/.bias
            yield f"text_projection.{sub}", t.to(torch.float32)
        rows = codec_w[torch.tensor(CODEC_SPECIAL_IDS, dtype=torch.long)].contiguous()
        print(f"[textfe] codec_special_embd rows={tuple(rows.shape)} ids={CODEC_SPECIAL_IDS}", flush=True)
        yield "codec_special_embd.weight", rows.to(torch.float32)

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> "Iterable[tuple[str, Tensor]]":
        if name.startswith("text_projection.") or name == "codec_special_embd.weight":
            yield name, data_torch            # custom names written verbatim
            return
        yield from super().modify_tensors(data_torch, name, bid)
