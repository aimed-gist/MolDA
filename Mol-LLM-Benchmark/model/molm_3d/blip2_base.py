"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os

import torch
import torch.nn as nn

from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from unicore.data import Dictionary
from .unimol import SimpleUniMolModel

# Get the directory where this file is located
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, bert_name=None):
        if bert_name is None:
            bert_name = 'allenai/scibert_scivocab_uncased'
        tokenizer = BertTokenizer.from_pretrained(bert_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, graph_width, cross_attention_freq=2):
        # Support both 'scibert' shorthand and full model name
        if model_name == 'scibert':
            bert_name = 'allenai/scibert_scivocab_uncased'
        else:
            bert_name = model_name

        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel.from_pretrained(
            bert_name, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_unimol_encoder(cls, args):
        # Use local unimol_dict.txt
        dict_path = os.path.join(_CURRENT_DIR, 'unimol_dict.txt')
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"UniMol dictionary not found at {dict_path}")

        dictionary = Dictionary.load(dict_path)
        dictionary.add_symbol("[MASK]", is_special=True)
        unimol_model = SimpleUniMolModel(args, dictionary)

        # Load UniMol pretrained weights if available
        unimol_ckpt_path = getattr(args, 'unimol_ckpt_path', None)
        if unimol_ckpt_path and os.path.exists(unimol_ckpt_path):
            ckpt = torch.load(unimol_ckpt_path, map_location=torch.device('cpu'))['model']
            missing_keys, unexpected_keys = unimol_model.load_state_dict(ckpt, strict=False)
            if missing_keys or unexpected_keys:
                print(f"[UniMol] Missing keys: {missing_keys}")
                print(f"[UniMol] Unexpected keys: {unexpected_keys}")
            print(f"[UniMol] Loaded pretrained weights from {unimol_ckpt_path}")
        else:
            print("[UniMol] No pretrained weights loaded - using random initialization")

        ln_graph = nn.LayerNorm(unimol_model.num_features)
        return unimol_model, ln_graph, dictionary

    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
