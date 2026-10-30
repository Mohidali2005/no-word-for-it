import json

import pytest
import torch
import transformers
from PIL import Image

from nwf import config
from nwf import models
from nwf import run_items
from nwf.paths import STIMULI

NEW_LIBRARY = int(transformers.__version__.split(".")[0])>=5


def tiny(monkeypatch,folder,name,mode):
    """Load a tiny stand in model with the real processor"""
    monkeypatch.setattr(models,"LOCAL_MODELS",str(folder))
    return models.load(name,mode)


def first_item(processor):
    """Encode the first item of the filled named set"""
    row = json.loads((STIMULI/"named_fill"/"items.jsonl").read_text(encoding="utf-8").splitlines()[0])
    images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
    return run_items.encode(processor,images,config.PROMPT,config.PREFILL)


@pytest.mark.parametrize("name",["qwen","gemma"])
def test_hooks_against_recorded_hidden_states(monkeypatch,tiny_models,name):
    model,processor = tiny(monkeypatch,tiny_models,name,"full")
    store = run_items.capture(model)
    deep = []
    if name=="qwen":
        for merger in model.model.visual.deepstack_merger_list:
            merger.register_forward_hook(lambda module,args,output:deep.append(output))
    enc = first_item(processor)
    store["positions"] = torch.arange(enc["input_ids"].shape[1])
    store["on"] = True
    with torch.no_grad():
        out = model(**enc,output_hidden_states=True)
    store["on"] = False
    hooked,recorded = store["states"],out.hidden_states
    layers = len(model.model.language_model.layers)
    assert len(hooked)==len(recorded)==layers+1
    if name=="qwen" and not NEW_LIBRARY:
        # older versions record the last qwen state before the final norm just like the hook
        assert torch.equal(hooked[layers],recorded[layers][0])
    else:
        # otherwise the library stores the last state after the final norm while the hook sees it before
        assert torch.allclose(model.model.language_model.norm(hooked[layers][None])[0],recorded[layers][0],atol=1e-5)
    image = enc["input_ids"][0]==model.config.image_token_id
    for s in range(layers):
        if name=="qwen" and s in (1,2,3) and NEW_LIBRARY:
            # newer versions record each early layer output before the extra vision features are added
            diff = hooked[s]-recorded[s][0]
            assert torch.allclose(diff[image],deep[s-1],atol=1e-5)
            assert diff[~image].abs().max()==0
        else:
            # older versions add those features in place so the recorded state already holds them
            assert torch.equal(hooked[s],recorded[s][0])


@pytest.mark.parametrize("name",["qwen","gemma"])
def test_hooks_do_not_change_the_output(monkeypatch,tiny_models,name):
    model,processor = tiny(monkeypatch,tiny_models,name,"full")
    enc = first_item(processor)
    with torch.no_grad():
        plain = model(**enc,logits_to_keep=1).logits
    store = run_items.capture(model)
    logits,states = run_items.forward(model,store,enc,list(range(5)))
    assert torch.equal(plain[0,-1],logits)
    assert len(states)==len(model.model.language_model.layers)+1


def test_gemma_fast_mode_keeps_language_part_fp32(monkeypatch,tiny_models):
    model,processor = tiny(monkeypatch,tiny_models,"gemma","fast")
    assert {p.dtype for p in model.model.vision_tower.parameters()}=={torch.float16}
    assert {p.dtype for p in model.model.language_model.parameters()}=={torch.float32}
    assert {p.dtype for p in model.model.multi_modal_projector.parameters()}=={torch.float32}
    full,_ = tiny(monkeypatch,tiny_models,"gemma","full")
    enc = first_item(processor)
    image = torch.nonzero(enc["input_ids"][0]==model.config.image_token_id).flatten().tolist()
    entering = []
    for m in (model,full):
        store = run_items.capture(m)
        entering.append(run_items.forward(m,store,enc,image)[1][0])
    cos = torch.nn.functional.cosine_similarity(entering[0],entering[1],dim=-1)
    assert entering[0].dtype==torch.float32 and entering[0].abs().max()>0
    assert cos.min()>0.999 and not torch.equal(entering[0],entering[1])
