import argparse
import json

import numpy as np

from nwf.analysis.behaviour import RESAMPLES
from nwf.analysis.behaviour import interval
from nwf.config import REMOVAL_DRAWS
from nwf.config import REMOVAL_KS
from nwf.config import REMOVAL_SETS
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS
from nwf.paths import STIMULI

MAIN_K = 16
# each named set is compared with the unnamed set of the same drawing style
CELLS = {"filled":("named_fill","unnamed_fill"),"line":("named_line","unnamed_line")}


def load(run,name,set_name):
    """Return per item whether each removal condition gives the right letter plus the shape cluster and the condition names"""
    items = {}
    for line in open(STIMULI/set_name/"items.jsonl",encoding="utf-8"):
        row = json.loads(line)
        items[row["id"]] = row
    folder = RESULTS/run/name/"removal"/set_name
    files = sorted(folder.glob("*.npz"))
    conditions = json.loads(files[0].with_suffix(".done.json").read_text(encoding="utf-8"))["conditions"]
    loaded = [np.load(f) for f in files]
    ids = np.concatenate([c["ids"] for c in loaded])
    rows = [items[i] for i in ids]
    answer = np.array(["ABCD".index(r["answer"]) for r in rows])
    hits = np.concatenate([c["logits"] for c in loaded]).argmax(-1)==answer[:,None]
    return hits,np.array([r["shapes"][r["ref_index"]] for r in rows]),conditions,ids


def drops(acc,conditions):
    """Return the accuracy drop of each caption removal and of its energy matched random control and their difference"""
    none = acc[...,conditions.index("none")]
    out = {}
    for k in REMOVAL_KS:
        caption = none-acc[...,conditions.index(f"caption{k}")]
        random = none-np.mean([acc[...,conditions.index(f"random{k}_{d}")] for d in range(REMOVAL_DRAWS)],0)
        out[k] = {"drop_caption":caption,"drop_random":random,"excess":caption-random}
    return out


def run_model(run,name):
    """Compute the removal drops of every core set and the named against unnamed difference of experiment four"""
    rng = np.random.default_rng(0)
    point,boot,table = {},{},{}
    for s in REMOVAL_SETS:
        hits,clusters,conditions,ids = load(run,name,s)
        groups = np.unique(clusters,return_inverse=True)[1]
        sums = np.zeros((groups.max()+1,hits.shape[1]))
        np.add.at(sums,groups,hits)
        counts = np.bincount(groups).astype(float)
        draws = rng.integers(0,len(sums),(RESAMPLES,len(sums)))
        weights = np.zeros((RESAMPLES,len(sums)))
        np.add.at(weights,(np.arange(RESAMPLES)[:,None],draws),1)
        acc = sums.sum(0)/counts.sum()
        acc_boot = (weights@sums)/(weights@counts)[:,None]
        point[s],boot[s] = drops(acc,conditions),drops(acc_boot,conditions)
        table[s] = {
            "items":len(hits),"clusters":len(sums),"first_id":str(ids[0]),"last_id":str(ids[-1]),
            "accuracy":{c:interval(acc[j],acc_boot[:,j]) for j,c in enumerate(conditions)},
        }
        for k in REMOVAL_KS:
            table[s][f"k{k}"] = {key:interval(point[s][k][key],boot[s][k][key]) for key in point[s][k]}

    effects = {}
    for k in REMOVAL_KS:
        effects[f"k{k}"] = {}
        # positive when taking out the caption directions hurts named items more than unnamed ones beyond the random control
        for cell,(a,b) in CELLS.items():
            effects[f"k{k}"][cell] = interval(point[a][k]["excess"]-point[b][k]["excess"],boot[a][k]["excess"]-boot[b][k]["excess"])
        effects[f"k{k}"]["main_mean_of_filled_and_line"] = interval(
            np.mean([point[a][k]["excess"]-point[b][k]["excess"] for a,b in CELLS.values()]),
            np.mean([boot[a][k]["excess"]-boot[b][k]["excess"] for a,b in CELLS.values()],0))
    return {"model":name,"main_k":MAIN_K,"sets":table,"effects":effects}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment four removal check for one model run")
    parser.add_argument("--run",default="kaggle_run_01/results")
    parser.add_argument("--name",required=True,help="folder of the model run for example qwen_fast")
    args = parser.parse_args()
    out = run_model(args.run,args.name)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS/f"e4_{args.name}.json").write_text(json.dumps(out,indent=1),encoding="utf-8")
    for s,row in out["sets"].items():
        print(f"{s:13s} n={row['items']} "+" ".join(f"{c}={a['value']:.3f}" for c,a in row["accuracy"].items()))
        for k in REMOVAL_KS:
            r = row[f"k{k}"]
            print(f"   k={k:2d} "+" ".join(f"{key}={v['value']:+.3f} [{v['low']:+.3f},{v['high']:+.3f}]" for key,v in r.items()))
    for k,row in out["effects"].items():
        print(f"effect {k}: "+" ".join(f"{cell}={v['value']:+.3f} [{v['low']:+.3f},{v['high']:+.3f}]" for cell,v in row.items()))
