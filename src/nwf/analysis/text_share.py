import argparse
import json

import numpy as np

from nwf.analysis.behaviour import DIFFERENCES
from nwf.analysis.behaviour import RESAMPLES
from nwf.analysis.probe import PROBE_VIEWS
from nwf.config import KS
from nwf.config import MODELS
from nwf.config import REMOVAL_FIRST_LAYER
from nwf.config import SETS
from nwf.config import VIEWS
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS
from nwf.paths import STIMULI

# the vocabulary and caption and prompt span text parts plus the image token pca as a non text baseline
DEFINITIONS = ["vocab","caption","sieve","image"]
# the share is not estimable when the full probe is this close to chance
MIN_ABOVE_CHANCE = 0.05
MAIN_K = 64


def load(run,name,set_name):
    """Return per item the hits of every named view and the mean hit over the random subspaces plus the shape cluster"""
    items = {}
    for line in open(STIMULI/set_name/"items.jsonl",encoding="utf-8"):
        row = json.loads(line)
        items[row["id"]] = row
    hits,clusters,test = [],[],[]
    for f in sorted((RESULTS/run/name/set_name).glob("*.npz")):
        chunk = np.load(f)
        rows = [items[i] for i in chunk["ids"]]
        answer = np.array(["ABCD".index(r["answer"]) for r in rows])
        named = chunk["scores"].argmax(-1)==answer[:,None,None,None]
        # the random answers are averaged over the subspaces for each size and for inside and outside
        random = (chunk["random"]==answer[:,None,None,None,None,None]).mean(3).reshape(*named.shape[:3],-1)
        hits.append(np.concatenate([named,random],-1))
        clusters += [r["shapes"][r["ref_index"]] for r in rows]
        # the released sets are all calibration items so they are used whole as a replication check
        test += [SETS[set_name]["kind"]=="rep" or not r["calibration"] for r in rows]
    test = np.array(test)
    return np.concatenate(hits)[test],np.array(clusters)[test]


def shares(acc,first):
    """Return accuracy inside and outside each part with its random baseline and the text share for every view and stream"""
    above = acc[...,0]-0.25
    out = {"full":acc[...,0]}
    for b in DEFINITIONS:
        for j,k in enumerate(KS):
            if b=="sieve":
                # the prompt span has no size k and its baseline is random subspaces of its own rank
                if k!=KS[0]:
                    continue
                label,inside,outside,size = "sieve",VIEWS.index("sieve"),VIEWS.index("sieve_res"),len(KS)
            else:
                label,inside,outside,size = f"{b}{k}",VIEWS.index(f"{b}{k}"),VIEWS.index(f"{b}{k}_res"),j
            random_in = acc[...,len(VIEWS)+2*size]
            with np.errstate(invalid="ignore",divide="ignore"):
                share = np.where(above>=MIN_ABOVE_CHANCE,(acc[...,inside]-random_in)/above,np.nan)
            out[label] = {
                "inside":acc[...,inside],"random_inside":random_in,
                "outside":acc[...,outside],"random_outside":acc[...,len(VIEWS)+2*size+1],
                "share":share,
                # the main statistic averages the share over the streams after the extra vision features are added
                "mean_share":np.nanmean(share[...,first:],-1),
            }
    return out


def nan_interval(point,samples):
    """Point value with the 2.5 and 97.5 percent bootstrap limits ignoring draws where the share is not estimable"""
    if np.isnan(point):
        return {"value":None,"low":None,"high":None}
    low,high = np.nanpercentile(samples,[2.5,97.5])
    return {"value":float(point),"low":float(low),"high":float(high)}


def summarise(hits,clusters,first,rng):
    """Return point values and bootstrap draws of the text shares of one set by resampling shape clusters"""
    groups = np.unique(clusters,return_inverse=True)[1]
    sums = np.zeros((groups.max()+1,hits[0].size),dtype=np.float32)
    np.add.at(sums,groups,hits.reshape(len(hits),-1))
    counts = np.bincount(groups).astype(np.float32)
    draws = rng.integers(0,len(sums),(RESAMPLES,len(sums)))
    weights = np.zeros((RESAMPLES,len(sums)),dtype=np.float32)
    np.add.at(weights,(np.arange(RESAMPLES)[:,None],draws),1)
    point = shares((sums.sum(0)/counts.sum()).reshape(hits.shape[1:]),first)
    with np.errstate(invalid="ignore"):
        boot = shares(((weights@sums)/(weights@counts)[:,None]).reshape(RESAMPLES,*hits.shape[1:]),first)
    # only the shares are kept from the draws to save memory
    boot = {key:{"share":value["share"],"mean_share":value["mean_share"]} for key,value in boot.items() if key!="full"}
    return point,boot,len(hits),len(sums)


def run_model(run,name):
    """Compute the text share of every set and the named against unnamed share differences of experiment three"""
    rng = np.random.default_rng(0)
    first = REMOVAL_FIRST_LAYER[MODELS[name.rsplit("_",1)[0]]["family"]]
    keep = {s for pair in DIFFERENCES.values() for s in pair}
    table,stats = {},{}
    for s in [s for s in SETS if (RESULTS/run/name/s).exists()]:
        point,boot,items,clusters = summarise(*load(run,name,s),first,rng)
        if s in keep:
            stats[s] = (point,boot)
        row = {"items":items,"clusters":clusters}
        for v,view in enumerate(PROBE_VIEWS):
            row[view] = {"full_by_stream":point["full"][v].round(4).tolist()}
            for label,p in point.items():
                if label=="full":
                    continue
                row[view][label] = {
                    "mean_share":nan_interval(p["mean_share"][v],boot[label]["mean_share"][:,v]),
                    "estimable_streams":int((~np.isnan(p["share"][v,first:])).sum()),
                    "share_by_stream":[nan_interval(a,c) for a,c in zip(p["share"][v],boot[label]["share"][:,v].T)],
                }
                for key in ("inside","random_inside","outside","random_outside"):
                    row[view][label][f"{key}_by_stream"] = p[key][v].round(4).tolist()
        table[s] = row

    labels = [key for key in stats["named_fill"][0] if key!="full"]
    diffs = {}
    for label,(a,b) in DIFFERENCES.items():
        diffs[label] = {}
        for v,view in enumerate(PROBE_VIEWS):
            diffs[label][view] = {}
            for part in labels:
                pa,ba = stats[a][0][part],stats[a][1][part]
                pb,bb = stats[b][0][part],stats[b][1][part]
                diffs[label][view][part] = {
                    "mean_share":nan_interval(pa["mean_share"][v]-pb["mean_share"][v],ba["mean_share"][:,v]-bb["mean_share"][:,v]),
                    "share_by_stream":[nan_interval(x,y) for x,y in zip(pa["share"][v]-pb["share"][v],(ba["share"][:,v]-bb["share"][:,v]).T)],
                }
    # the two within filled and within line differences averaged into one number
    means = {}
    for v,view in enumerate(PROBE_VIEWS):
        means[view] = {}
        for part in labels:
            point = sum(stats[a][0][part]["mean_share"][v]-stats[b][0][part]["mean_share"][v] for a,b in (("named_fill","unnamed_fill"),("named_line","unnamed_line")))/2
            boot = sum(stats[a][1][part]["mean_share"][:,v]-stats[b][1][part]["mean_share"][:,v] for a,b in (("named_fill","unnamed_fill"),("named_line","unnamed_line")))/2
            means[view][part] = {"mean_share":nan_interval(point,boot)}
    diffs["e1b_main_mean_of_filled_and_line"] = means
    return {"model":name,"first_stream":first,"main_k":MAIN_K,"sets":table,"differences":diffs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment three text share for one model run")
    parser.add_argument("--run",default="kaggle_run_01/results")
    parser.add_argument("--name",required=True,help="folder of the model run for example qwen_fast")
    args = parser.parse_args()
    out = run_model(args.run,args.name)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS/f"e3_{args.name}.json").write_text(json.dumps(out,indent=1),encoding="utf-8")
    parts = [f"vocab{MAIN_K}",f"caption{MAIN_K}","sieve",f"image{MAIN_K}"]
    for view in PROBE_VIEWS:
        print(f"{view}: mean share over streams from {out['first_stream']}")
        for s,row in out["sets"].items():
            cells = [f"{p}={row[view][p]['mean_share']['value']:+.3f} [{row[view][p]['mean_share']['low']:+.3f},{row[view][p]['mean_share']['high']:+.3f}]" for p in parts]
            print(f"  {s:13s} "+" ".join(cells))
        for label,row in out["differences"].items():
            cells = [f"{p}={row[view][p]['mean_share']['value']:+.3f} [{row[view][p]['mean_share']['low']:+.3f},{row[view][p]['mean_share']['high']:+.3f}]" for p in parts]
            print(f"  {label}: "+" ".join(cells))
