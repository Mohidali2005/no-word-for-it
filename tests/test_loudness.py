import json
import runpy
import sys

import numpy as np
import pytest
import statsmodels.api as sm

from nwf import config
from nwf import paths

ITEMS = 160
CALIBRATION = 40
STREAMS = 6
RANDOMS = 5
# the loud directions lift the margin of unnamed items by this much in each fake model
BOOSTS = {"qwen_fast":2.0,"qwen_names_fast":1.0}


def build(root,name,boost,seed=1):
    """Write fake items and result chunks where behaviour follows the margin only and unnamed probes gain from the loud removal"""
    rng = np.random.default_rng(seed)
    truth = {}
    for set_name,spec in config.SETS.items():
        named = spec["named"]
        pool = [f"{set_name}_shape{j}" for j in range(8 if named else 20)]
        rows = []
        for i in range(ITEMS):
            rows.append({
                "id":f"{set_name}_{i:04d}",
                "answer":"ABCD"[rng.integers(4)],
                "shapes":[str(s) for s in rng.choice(pool,4,replace=False)],
                "ref_index":int(rng.integers(4)),
                "calibration":bool(spec["kind"]=="rep" or i<CALIBRATION),
                "named":named,
                "filled":spec["filled"],
                "complexity":[float(c) for c in rng.uniform(1,200,4)],
                "lookalike":float(rng.uniform(0,1)),
            })
        folder = root/"stimuli"/set_name
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"items.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
        answer = np.array(["ABCD".index(r["answer"]) for r in rows])
        margin = rng.normal(2.0 if named else 0.3,1.0,ITEMS)
        correct = margin+rng.logistic(0,0.5,ITEMS)>0.3
        gain = 0.0 if named else boost
        # the wrong targets sit just below one half so the right one wins exactly when its margin is positive
        base = np.tile(0.5-0.001*np.arange(4),(ITEMS,1)).astype(np.float32)
        full,loud = base.copy(),base.copy()
        for i in range(ITEMS):
            wrong = [t for t in range(4) if t!=answer[i]]
            full[i,wrong] = base[i,:3]
            loud[i,wrong] = base[i,:3]
            full[i,answer[i]] = 0.5+0.01*margin[i]
            loud[i,answer[i]] = 0.5+0.01*(margin[i]+gain)
        scores = np.broadcast_to(full[:,None,None,None,:],(ITEMS,2,STREAMS,len(config.VIEWS),4)).copy()
        for k in config.KS:
            scores[:,:,:,config.VIEWS.index(f"image{k}_res"),:] = loud[:,None,None,:]
        # outside every random subspace the answer is the full one and inside it is always wrong so mixing the two parts shows up
        random = np.zeros((ITEMS,2,STREAMS,RANDOMS,len(config.KS)+1,2),dtype=np.uint8)
        random[...,1] = full.argmax(-1)[:,None,None,None,None]
        random[...,0] = ((answer+1)%4)[:,None,None,None,None]
        logits = np.zeros((ITEMS,4),dtype=np.float32)
        logits[np.arange(ITEMS),np.where(correct,answer,(answer+1)%4)] = 5.0
        out = root/"results"/"run"/name/set_name
        out.mkdir(parents=True,exist_ok=True)
        for first in (0,ITEMS//2):
            cut = slice(first,first+ITEMS//2)
            ids = np.array([r["id"] for r in rows[cut]])
            np.savez(out/f"{first:04d}.npz",ids=ids,logits=logits[cut],scores=scores[cut],random=random[cut])
        test = np.array([not r["calibration"] or spec["kind"]=="rep" for r in rows])
        truth[set_name] = {"margin":margin,"correct":correct,"test":test,"gain":gain,"rows":rows}
    return truth


def run_script(monkeypatch,root,names):
    """Run the loudness script on the fake folders and return its saved result"""
    monkeypatch.setattr(paths,"STIMULI",root/"stimuli")
    monkeypatch.setattr(paths,"RESULTS",root/"results")
    monkeypatch.setattr(paths,"ANALYSIS",root/"analysis")
    monkeypatch.setattr(sys,"argv",["loudness","--run","run","--names"]+names)
    runpy.run_module("nwf.analysis.loudness",run_name="__main__")
    return json.loads((root/"analysis"/f"loudness_{'_vs_'.join(names)}.json").read_text(encoding="utf-8"))


def test_loudness_gains_match_the_planted_values(monkeypatch,tmp_path):
    truth = {name:build(tmp_path,name,boost) for name,boost in BOOSTS.items()}
    out = run_script(monkeypatch,tmp_path,list(BOOSTS))
    assert out["status"]=="exploratory"
    for name in BOOSTS:
        for set_name,t in truth[name].items():
            m = t["margin"][t["test"]]
            expected = (m+t["gain"]>0).mean()-(m>0).mean()
            row = out["models"][name]["sets"][set_name]
            for view in ("raw","massive_removed"):
                assert row[view]["excess_at_layer"]["value"]==pytest.approx(expected,abs=1e-12)
                assert row[view]["excess_mean_over_streams"]["value"]==pytest.approx(expected,abs=1e-12)
            if set_name.startswith("named") or set_name=="rep_named":
                assert expected==0
    # the second model gains less so the change is the difference of the two planted gains
    for set_name,t in truth["qwen_fast"].items():
        m = t["margin"][t["test"]]
        expected = (m+BOOSTS["qwen_names_fast"]*(t["gain"]>0)>0).mean()-(m+t["gain"]>0).mean()
        assert out["change_in_excess_raw_view"]["sets"][set_name]["value"]==pytest.approx(expected,abs=1e-12)
    diffs = out["models"]["qwen_fast"]["differences"]
    assert set(diffs)=={"unnamed_line_minus_named_fill","rep_squig30_minus_rep_named","unnamed_fill_minus_named_fill","unnamed_line_minus_named_line","squig_n50_minus_squig_n5"}
    assert diffs["unnamed_line_minus_named_fill"]["raw"]["at_layer"]["value"]>0


def test_margin_regression_matches_a_direct_fit(monkeypatch,tmp_path):
    truth = build(tmp_path,"qwen_fast",2.0)
    out = run_script(monkeypatch,tmp_path,["qwen_fast"])
    reg = out["models"]["qwen_fast"]["regression"]
    core = ["named_fill","named_line","unnamed_fill","unnamed_line"]
    rows = [r for s in core for r,keep in zip(truth[s]["rows"],truth[s]["test"]) if keep]
    y = np.concatenate([truth[s]["correct"][truth[s]["test"]] for s in core]).astype(float)
    # rebuilding the margin from the stored float32 scores so it matches the script exactly
    margin = np.concatenate([np.float32(0.5+0.01*truth[s]["margin"][truth[s]["test"]])-np.float32(0.5) for s in core])
    z = (margin-margin.mean())/margin.std()
    X = np.column_stack([[r["named"] for r in rows],[r["filled"] for r in rows],[np.log(r["complexity"][r["ref_index"]]) for r in rows],[r["lookalike"] for r in rows]]).astype(float)
    groups = np.unique([r["shapes"][r["ref_index"]] for r in rows],return_inverse=True)[1]
    plain = sm.Logit(y,sm.add_constant(X)).fit(disp=0,cov_type="cluster",cov_kwds={"groups":groups})
    full = sm.Logit(y,sm.add_constant(np.column_stack([X,z]))).fit(disp=0,cov_type="cluster",cov_kwds={"groups":groups})
    assert reg["without_margin"]["named"]["coef"]==pytest.approx(round(plain.params[1],4),abs=1e-9)
    assert reg["with_margin"]["named"]["coef"]==pytest.approx(round(full.params[1],4),abs=1e-9)
    assert reg["with_margin"]["margin"]["coef"]==pytest.approx(round(full.params[5],4),abs=1e-9)
    # behaviour follows the margin only so the name effect shrinks once the margin is in
    assert abs(reg["with_margin"]["named"]["coef"])<reg["without_margin"]["named"]["coef"]/2
    for s in core:
        assert reg["margin_within_set"][s]["coef"]>0
    assert reg["items"]==len(y)


def test_runs_that_scored_different_items_are_refused(monkeypatch,tmp_path):
    build(tmp_path,"qwen_fast",2.0)
    build(tmp_path,"qwen_names_fast",1.0)
    # renaming the ids of one chunk so the two runs no longer line up
    chunk = tmp_path/"results"/"run"/"qwen_names_fast"/"named_fill"/"0080.npz"
    saved = dict(np.load(chunk))
    saved["ids"] = saved["ids"][::-1]
    np.savez(chunk,**saved)
    with pytest.raises((AssertionError,KeyError)):
        run_script(monkeypatch,tmp_path,["qwen_fast","qwen_names_fast"])
