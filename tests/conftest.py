import os

import pytest
import torch
from transformers import AutoConfig
from transformers import Gemma3ForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration

from nwf import config

# using the plain image backend so tests never try to open a window
os.environ["MPLBACKEND"] = "Agg"


def tiny_config(name):
    """Shrink the real config of a model while keeping its vocabulary image sizes and wiring"""
    spec = config.MODELS[name]
    real = AutoConfig.from_pretrained(spec["repo"],revision=spec["revision"]).to_dict()
    text,vision = real["text_config"],real["vision_config"]
    if spec["family"]=="qwen":
        # qwen stays as wide as the largest subspace so the full run can be tested with the real settings
        text.update(hidden_size=256,intermediate_size=512,num_hidden_layers=5,num_attention_heads=4,num_key_value_heads=2,head_dim=64)
        # the rotary sections must add up to half the smaller head size
        for key in ("rope_scaling","rope_parameters"):
            if text.get(key) and "mrope_section" in text[key]:
                text[key]["mrope_section"] = [12,10,10]
        vision.update(depth=4,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=256,deepstack_visual_indexes=[1,2,3])
    else:
        text.update(hidden_size=64,intermediate_size=128,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=32,query_pre_attn_scalar=32)
        text.pop("layer_types",None)
        vision.update(hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2)
    return type(AutoConfig.from_pretrained(spec["repo"],revision=spec["revision"])).from_dict(real)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Save tiny random qwen and gemma models built from the real configs and return their folder"""
    folder = tmp_path_factory.mktemp("models")
    for seed,(name,cls) in enumerate([("qwen",Qwen3VLForConditionalGeneration),("gemma",Gemma3ForConditionalGeneration)]):
        torch.manual_seed(seed)
        model = cls(tiny_config(name)).eval()
        if name=="gemma":
            # the library starts the image projector at zero which would hide every image token
            torch.nn.init.normal_(model.model.multi_modal_projector.mm_input_projection_weight,std=0.1)
        model.save_pretrained(folder/name)
    return folder
