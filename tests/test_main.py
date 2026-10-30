import json
import time

import numpy as np

from nwf import config
from nwf import main
from nwf import precision
from test_run_items import CAPTIONS


def use_folders(monkeypatch,tiny_models,tmp_path):
    """Point the child processes and this one at tiny models and temporary folders"""
    (tmp_path/"captions.txt").write_text(CAPTIONS,encoding="utf-8")
    monkeypatch.setenv("NWF_LOCAL_MODELS",str(tiny_models))
    monkeypatch.setenv("NWF_RESULTS",str(tmp_path/"results"))
    monkeypatch.setenv("NWF_SCRATCH",str(tmp_path/"scratch"))
    monkeypatch.setenv("NWF_CAPTIONS",str(tmp_path/"captions.txt"))
    for module in (main,precision):
        monkeypatch.setattr(module,"RESULTS",tmp_path/"results")
    monkeypatch.setattr(precision,"SCRATCH",tmp_path/"scratch")


def test_smoke_run_end_to_end_with_real_settings(monkeypatch,tiny_models,tmp_path):
    use_folders(monkeypatch,tiny_models,tmp_path)
    main.kaggle(["qwen:full","qwen:fast"],True,time.time()+3600)
    results = tmp_path/"results"
    report = json.loads((results/"run_report.json").read_text(encoding="utf-8"))
    assert [r["step"] for r in report]==["qwen:full","qwen:full:removal","qwen:fast","qwen:fast:removal"]
    assert [r["exit_code"] for r in report]==[0,0,0,0]
    for set_name in config.REMOVAL_SETS:
        assert np.load(results/"qwen_fast"/"removal"/set_name/"0000.npz")["logits"].shape[0]==5
    for mode in ("full","fast"):
        for set_name in config.SETS:
            saved = np.load(results/f"qwen_{mode}"/set_name/"0000.npz")
            assert len(saved["ids"])==config.SMOKE_ITEMS
            assert saved["random"].shape[3]==config.RANDOM_SUBSPACES
    checks = json.loads((results/"qwen_full"/"smoke_checks.json").read_text(encoding="utf-8"))
    assert len(checks["answers"])==config.GENERATE_ITEMS*len(config.SETS)
    assert not (results/"qwen_fast"/"smoke_checks.json").exists()
    verdict = json.loads((results/"precision_qwen.json").read_text(encoding="utf-8"))
    assert verdict["items"]==config.SMOKE_ITEMS*len(config.SETS) and verdict["no_nan_or_inf"]


def test_nothing_starts_after_the_deadline(monkeypatch,tiny_models,tmp_path):
    use_folders(monkeypatch,tiny_models,tmp_path)
    main.kaggle(["qwen:full"],False,time.time()-1)
    assert not (tmp_path/"results"/"qwen_full").exists()
