import json
from functools import partial

import numpy as np
import torch
from PIL import Image

from nwf import config
from nwf import removal
from nwf import run_items
from nwf.paths import STIMULI
from test_run_items import run


def test_removal_takes_directions_out_from_the_first_layer_on(monkeypatch,tiny_models,tmp_path):
    model,processor,store = run(monkeypatch,tiny_models,tmp_path,"qwen","full","a")
    row = json.loads((STIMULI/"named_fill"/"items.jsonl").read_text(encoding="utf-8").splitlines()[0])
    images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
    enc = run_items.encode(processor,images,config.PROMPT,config.PREFILL)
    image = torch.nonzero(enc["input_ids"][0]==model.config.image_token_id).flatten()
    text = torch.nonzero(enc["input_ids"][0]!=model.config.image_token_id).flatten()
    everything = list(range(enc["input_ids"].shape[1]))
    _,before = run_items.forward(model,store,enc,everything)
    layers = model.model.language_model.layers
    d = model.config.text_config.hidden_size
    directions = [torch.from_numpy(np.linalg.qr(np.random.default_rng(s).standard_normal((d,8)))[0]).float() for s in range(len(layers)+1)]
    # the removal hooks go in front of the capture hooks so the capture sees what each layer really reads
    for s,layer in enumerate(layers):
        if s>=3:
            layer.register_forward_pre_hook(partial(removal.cut,positions=image,directions=directions[s]),prepend=True)
    _,after = run_items.forward(model,store,enc,everything)
    for s in range(len(layers)):
        inside = (after[s][image]@directions[s]).abs().max()
        if s>=3:
            assert inside<1e-4
        else:
            assert torch.equal(after[s],before[s]) and inside>1e-3
    # the text tokens entering the first changed layer are untouched
    assert torch.equal(after[3][text],before[3][text])


def test_removal_run_saves_every_condition(monkeypatch,tiny_models,tmp_path):
    model,processor,store = run(monkeypatch,tiny_models,tmp_path,"qwen","full","a")
    monkeypatch.setattr(removal,"RESULTS",tmp_path/"a")
    removal.run(model,processor,"qwen","full","qwen",2,None)
    found,sizes = removal.conditions("qwen","full",model.config.text_config.hidden_size)
    assert list(found)==["none","caption2","random2_0","random2_1","random2_2","caption16","random16_0","random16_1","random16_2"]
    share = np.load(tmp_path/"a"/"qwen_full"/"bases.npz")["caught_share"]
    d = model.config.text_config.hidden_size
    for j,k in enumerate(config.REMOVAL_KS):
        assert sizes[k]==[int(min(d,max(k,round(float(x)*d)))) for x in share[:,j]]
        assert all(found[f"random{k}_0"][s].shape==(d,sizes[k][s]) for s in range(len(share)))
    for set_name in config.REMOVAL_SETS:
        saved = np.load(tmp_path/"a"/"qwen_full"/"removal"/set_name/"0000.npz")
        assert saved["logits"].shape==(2,len(found),4) and np.isfinite(saved["logits"]).all()
        # with nothing removed the letters equal those of the main run
        main_run = np.load(tmp_path/"a"/"qwen_full"/set_name/"0000.npz")
        assert np.allclose(saved["logits"][:,0],main_run["logits"],atol=1e-5)
        assert not np.allclose(saved["logits"][:,0],saved["logits"][:,1])
