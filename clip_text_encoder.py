import os
import sys
import json
import torch

import torch.nn.functional as F

from transformers import CLIPTokenizer
from safetensors.torch import load_file
from accelerate import init_empty_weights


class CLIPTextEncoder:

    def __init__(self, model_path=None):

        self.model_path = model_path or os.environ.get(
            "CLIP_MODEL_PATH",
            "/content/models/eva_clip_text"
        )

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        sys.path.insert(
            0, os.path.dirname(self.model_path)
        )

        from eva_clip_text.configuration_evaclip import (
            EvaCLIPTextConfig
        )

        from eva_clip_text.modeling_evaclip import (
            EvaCLIPTextModelWithProjection
        )

        print("Loading EVA-CLIP text encoder...")

        with open(
            os.path.join(self.model_path, "config.json")
        ) as f:
            full_config = json.load(f)

        text_config = EvaCLIPTextConfig(
            **full_config["text_config"]
        )

        # Important: use the shared CLIP projection
        # dimension, not the text config's default.
        text_config.projection_dim = (
            full_config["projection_dim"]
        )

        with init_empty_weights():
            self.model = (
                EvaCLIPTextModelWithProjection(
                    text_config
                )
            )

        state_dict = load_file(
            os.path.join(
                self.model_path,
                "model.safetensors"
            )
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
            assign=True
        )

        del state_dict

        # Reinitialize the non-persistent position buffer.
        self.model.text_model.embeddings.position_ids = (
            torch.arange(
                text_config.max_position_embeddings
            ).expand(1, -1)
        )

        if self.device == "cpu":
            self.model = self.model.float()

        self.model = self.model.to(self.device).eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        print("EVA-CLIP text encoder loaded")

    @torch.inference_mode()
    def encode(self, texts):

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(self.device)

        output = self.model(
            input_ids=input_ids
        )

        embeddings = output.text_embeds.float()

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1
        )

        return embeddings.cpu().numpy()