from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers import Gemma3ForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration

from nwf import config
from nwf.paths import LOCAL_MODELS


def to_half(module,args,kwargs):
    """Cast the pixels going into the vision tower to fp16"""
    kwargs["pixel_values"] = kwargs["pixel_values"].half()
    return args,kwargs


def to_float(module,args,output):
    """Cast the vision tower output back to fp32 before the projector sees it"""
    output.last_hidden_state = output.last_hidden_state.float()
    return output


def load(name,mode):
    """Load a model in eval mode with its processor either fully in fp32 or in the faster candidate mode"""
    spec = config.MODELS[name]
    family = spec["family"]
    base = config.MODELS[spec.get("processor",name)]
    extra = {"do_pan_and_scan":True} if family=="gemma" else {}
    processor = AutoProcessor.from_pretrained(base["repo"],revision=base["revision"],use_fast=config.FAMILIES[family]["use_fast"],**extra)
    source,revision = spec["repo"],spec["revision"]
    if LOCAL_MODELS:
        source,revision = Path(LOCAL_MODELS)/name,None
    gpus = torch.cuda.device_count()
    if family=="qwen":
        dtype = torch.float16 if mode=="fast" else torch.float32
        model = Qwen3VLForConditionalGeneration.from_pretrained(source,revision=revision,dtype=dtype,attn_implementation="sdpa",device_map={"":0} if gpus else None)
    else:
        # gemma overflows in fp16 so its language part stays fp32 and is split over both gpus
        model = Gemma3ForConditionalGeneration.from_pretrained(source,revision=revision,dtype=torch.float32,attn_implementation="sdpa",device_map="auto" if gpus else None)
        if mode=="fast":
            vision = model.model.vision_tower
            vision.half()
            vision.register_forward_pre_hook(to_half,with_kwargs=True)
            vision.register_forward_hook(to_float)
    model.eval()
    return model,processor
