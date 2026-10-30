import json

import numpy as np
import pytest
import torch
from PIL import Image

from nwf import config
from nwf import maxsim
from nwf import models
from nwf import precision
from nwf import run_items
from nwf.paths import STIMULI

CAPTIONS = "A cat sitting on a red sofa.\nTwo dogs run along a sandy beach.\nA plate of food on a wooden table.\n"


def run(monkeypatch,tiny_models,tmp_path,name,mode,folder):
    """Run the bases and the smoke scoring of a tiny model into a results folder"""
    monkeypatch.setattr(models,"LOCAL_MODELS",str(tiny_models))
    for module in (run_items,precision):
        monkeypatch.setattr(module,"RESULTS",tmp_path/folder)
        monkeypatch.setattr(module,"SCRATCH",tmp_path/folder/"scratch")
    (tmp_path/"captions.txt").write_text(CAPTIONS,encoding="utf-8")
    monkeypatch.setattr(run_items,"CAPTIONS",tmp_path/"captions.txt")
    # the tiny models are only 64 wide so the subspace sizes shrink with them
    monkeypatch.setattr(config,"KS",[4,16,64])
    monkeypatch.setattr(config,"RANDOM_SUBSPACES",5)
    model,processor = models.load(name,mode)
    store = run_items.capture(model)
    family = config.MODELS[name]["family"]
    run_items.fit_bases(model,processor,store,name,mode,family,3,1)
    run_items.score_items(model,processor,store,name,mode,family,2,True)
    return model,processor,store


@pytest.mark.parametrize("name",["qwen","gemma"])
def test_smoke_pipeline_on_tiny_models(monkeypatch,tiny_models,tmp_path,name):
    model,processor,store = run(monkeypatch,tiny_models,tmp_path,name,"full","a")
    out = tmp_path/"a"/f"{name}_full"
    streams = len(model.model.language_model.layers)+1
    d = model.config.text_config.hidden_size
    bases = np.load(out/"bases.npz")
    assert bases["vocab"].shape==(d,64) and bases["caption"].shape==(streams,d,64) and bases["image"].shape==(streams,d,64)
    assert np.allclose(bases["image"][1].T@bases["image"][1],np.eye(64),atol=1e-4)
    for set_name in config.SETS:
        saved = np.load(out/set_name/"0000.npz")
        assert len(saved["ids"])==2
        assert saved["logits"].shape==(2,4)
        assert saved["scores"].shape==(2,2,streams,len(config.VIEWS),4)
        assert saved["random"].shape==(2,2,streams,5,4,2)
        assert saved["pooled"].shape==(2,streams,5,d)
        assert all(np.isfinite(saved[k].astype(np.float64)).all() for k in ("logits","scores","pooled","norms"))
    # recomputing one item by hand gives the same letters and full space probe scores
    row = json.loads((STIMULI/"named_line"/"items.jsonl").read_text(encoding="utf-8").splitlines()[1])
    images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
    enc = run_items.encode(processor,images,config.PROMPT,config.PREFILL)
    with torch.no_grad():
        logits = model(**enc).logits[0,-1]
    saved = np.load(out/"named_line"/"0000.npz")
    assert np.allclose(saved["logits"][1],logits[run_items.letter_ids(processor.tokenizer)].numpy(),atol=1e-5)
    regions,prompt = run_items.locate(enc,row,config.MODELS[name]["family"],processor.tokenizer,images)
    _,states = run_items.forward(model,store,enc,sum(regions,[]))
    cuts = np.cumsum([len(r) for r in regions])[:-1].tolist()
    for s in (0,streams-1):
        parts = torch.tensor_split(states[s],cuts)
        assert np.allclose(saved["scores"][1,0,s,0],maxsim.maxsim(parts[0],list(parts[1:])).numpy(),atol=1e-5)
    assert saved["sizes"][1].tolist()==[len(r) for r in regions]+[len(prompt)]


def test_resume_skips_finished_chunks_and_reruns_are_identical(monkeypatch,tiny_models,tmp_path):
    run(monkeypatch,tiny_models,tmp_path,"qwen","full","a")
    first = tmp_path/"a"/"qwen_full"/"unnamed_line"/"0000.npz"
    stamp = first.stat().st_mtime_ns
    model,processor,store = run(monkeypatch,tiny_models,tmp_path,"qwen","full","a")
    assert first.stat().st_mtime_ns==stamp
    run(monkeypatch,tiny_models,tmp_path,"qwen","full","b")
    for set_name in config.SETS:
        a = np.load(tmp_path/"a"/"qwen_full"/set_name/"0000.npz")
        b = np.load(tmp_path/"b"/"qwen_full"/set_name/"0000.npz")
        assert all(np.array_equal(a[k],b[k]) for k in a.files)


@pytest.mark.parametrize("name",["qwen","gemma"])
def test_precision_check_on_tiny_models(monkeypatch,tiny_models,tmp_path,name):
    run(monkeypatch,tiny_models,tmp_path,name,"full","a")
    run(monkeypatch,tiny_models,tmp_path,name,"fast","a")
    verdict = precision.compare(name)
    assert verdict["items"]==2*len(config.SETS)
    assert verdict["no_nan_or_inf"]
    assert verdict["entering_cos_min"]>0.99
    saved = json.loads((tmp_path/"a"/f"precision_{name}.json").read_text(encoding="utf-8"))
    assert saved["passed"]==verdict["passed"]


def test_smoke_checks_write_answers_and_names(monkeypatch,tiny_models,tmp_path):
    model,processor,store = run(monkeypatch,tiny_models,tmp_path,"qwen","full","a")
    monkeypatch.setattr(config,"SETS",{k:v for k,v in config.SETS.items() if k in ("named_fill","unnamed_line")})
    run_items.smoke_checks(model,processor,"qwen","full",1)
    saved = json.loads((tmp_path/"a"/"qwen_full"/"smoke_checks.json").read_text(encoding="utf-8"))
    assert sorted(saved["answers"])==["named_fill_0000","unnamed_line_0000"]
    assert len(saved["naming"])==8+8+10
