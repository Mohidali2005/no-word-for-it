import argparse
import json

import numpy as np

from nwf.analysis.behaviour import DIFFERENCES
from nwf.analysis.behaviour import RESAMPLES
from nwf.analysis.behaviour import interval
from nwf.config import SETS
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS
from nwf.paths import STIMULI

# every item holds probe scores on the raw vectors and on the vectors with the massive dimensions zeroed
PROBE_VIEWS = ["raw","massive_removed"]


def load(run,name,set_name):
    """Return per item the behaviour hit the probe hits at every stream of both views and the shape cluster"""
    items = {}
    for line in open(STIMULI/set_name/"items.jsonl",encoding="utf-8"):
        row = json.loads(line)
        items[row["id"]] = row
    loaded = [np.load(f) for f in sorted((RESULTS/run/name/set_name).glob("*.npz"))]
    ids = np.concatenate([c["ids"] for c in loaded])
    rows = [items[i] for i in ids]
    answer = np.array(["ABCD".index(r["answer"]) for r in rows])
    pred = np.concatenate([c["logits"] for c in loaded]).argmax(-1)
    # the full view is the first named view and the probe picks the target with the highest score
    probe = np.concatenate([c["scores"][:,:,:,0].argmax(-1) for c in loaded])
    calibration = np.array([r["calibration"] for r in rows])
    return {
        "correct":pred==answer,
        "probe":probe==answer[:,None,None],
        "cluster":np.array([r["shapes"][r["ref_index"]] for r in rows]),
        "calibration":calibration,
        # the released sets are all calibration items so they are used whole as a replication check
        "test":np.ones(len(rows),dtype=bool) if SETS[set_name]["kind"]=="rep" else ~calibration,
    }


def summarise(d,rng):
    """Return point values and bootstrap draws of the probe and behaviour rates of one set over shape clusters"""
    test = d["test"]
    correct,probe,cluster = d["correct"][test],d["probe"][test],d["cluster"][test]
    clusters = np.unique(cluster,return_inverse=True)[1]
    # per cluster sums of items behaviour hits probe hits and items hit by both
    sums = np.zeros((clusters.max()+1,2+2*probe[0].size))
    np.add.at(sums,clusters,np.column_stack([np.ones(len(correct)),correct,probe.reshape(len(probe),-1),(probe&correct[:,None,None]).reshape(len(probe),-1)]))
    draws = rng.integers(0,len(sums),(RESAMPLES,len(sums)))
    weights = np.zeros((RESAMPLES,len(sums)))
    np.add.at(weights,(np.arange(RESAMPLES)[:,None],draws),1)
    out = {}
    for label,total in (("point",sums.sum(0)[None]),("boot",weights@sums)):
        n,hit = total[:,0],total[:,1]
        both_hit = total[:,2:].reshape(len(total),2,2,-1)
        probe_hit,both = both_hit[:,0],both_hit[:,1]
        with np.errstate(invalid="ignore",divide="ignore"):
            out[label] = {
                "behaviour":hit/n,
                "probe":probe_hit/n[:,None,None],
                "behaviour_given_probe":both/probe_hit,
                "gap":(probe_hit-hit[:,None,None])/n[:,None,None],
            }
    out["items"] = int(len(correct))
    out["clusters"] = int(len(sums))
    return out


def curve(point,boot):
    """Point values with bootstrap limits for every stream index"""
    return [interval(p,b) for p,b in zip(point,boot.T)]


def run_model(run,name):
    """Compute the probe layer the probe and behaviour rates and the named against unnamed differences of experiment two"""
    rng = np.random.default_rng(0)
    data = {s:load(run,name,s) for s in SETS if (RESULTS/run/name/s).exists()}
    # the probe layer is the stream with the best probe accuracy over the calibration items of all sets pooled
    pooled = np.concatenate([d["probe"][d["calibration"]] for d in data.values()]).mean(0)
    layers = pooled.argmax(-1)
    stats = {s:summarise(d,rng) for s,d in data.items()}

    table = {}
    for s,st in stats.items():
        p,b = st["point"],st["boot"]
        row = {"items":st["items"],"clusters":st["clusters"],"behaviour":interval(p["behaviour"][0],b["behaviour"])}
        for v,view in enumerate(PROBE_VIEWS):
            layer = layers[v]
            row[view] = {
                "probe":interval(p["probe"][0,v,layer],b["probe"][:,v,layer]),
                "behaviour_given_probe":interval(p["behaviour_given_probe"][0,v,layer],b["behaviour_given_probe"][:,v,layer]),
                "gap":interval(p["gap"][0,v,layer],b["gap"][:,v,layer]),
                "probe_by_stream":curve(p["probe"][0,v],b["probe"][:,v]),
            }
        table[s] = row

    diffs = {}
    for label,(a,b) in DIFFERENCES.items():
        diffs[label] = {}
        pa,ba,pb,bb = stats[a]["point"],stats[a]["boot"],stats[b]["point"],stats[b]["boot"]
        for v,view in enumerate(PROBE_VIEWS):
            # named minus unnamed for behaviour among probe hits and unnamed minus named for the unused gap so both are positive under the hypothesis
            given = (pa["behaviour_given_probe"][:,v]-pb["behaviour_given_probe"][:,v],ba["behaviour_given_probe"][:,v]-bb["behaviour_given_probe"][:,v])
            gap = (pb["gap"][:,v]-pa["gap"][:,v],bb["gap"][:,v]-ba["gap"][:,v])
            diffs[label][view] = {
                "e2b_behaviour_given_probe":interval(given[0][0,layers[v]],given[1][:,layers[v]]),
                "e2c_gap":interval(gap[0][0,layers[v]],gap[1][:,layers[v]]),
                "e2b_by_stream":curve(given[0][0],given[1]),
                "e2c_by_stream":curve(gap[0][0],gap[1]),
            }

    # the two within filled and within line differences averaged into one number
    means = {}
    for v,view in enumerate(PROBE_VIEWS):
        layer = layers[v]
        means[view] = {}
        for key,sign in (("behaviour_given_probe",1),("gap",-1)):
            point = sum(sign*(stats[a]["point"][key][0,v,layer]-stats[b]["point"][key][0,v,layer]) for a,b in (("named_fill","unnamed_fill"),("named_line","unnamed_line")))/2
            boot = sum(sign*(stats[a]["boot"][key][:,v,layer]-stats[b]["boot"][key][:,v,layer]) for a,b in (("named_fill","unnamed_fill"),("named_line","unnamed_line")))/2
            means[view][key] = interval(point,boot)
    diffs["e1b_main_mean_of_filled_and_line"] = means

    probe_layer = {view:{"stream":int(layers[v]),"pooled_calibration_probe_by_stream":pooled[v].round(4).tolist()} for v,view in enumerate(PROBE_VIEWS)}
    return {"model":name,"probe_layer":probe_layer,"sets":table,"differences":diffs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment two encoded but unused for one model run")
    parser.add_argument("--run",default="kaggle_run_01/results")
    parser.add_argument("--name",required=True,help="folder of the model run for example qwen_fast")
    args = parser.parse_args()
    out = run_model(args.run,args.name)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS/f"e2_{args.name}.json").write_text(json.dumps(out,indent=1),encoding="utf-8")
    for view in PROBE_VIEWS:
        print(f"{view}: probe layer {out['probe_layer'][view]['stream']}")
        for s,row in out["sets"].items():
            r = row[view]
            print(f"  {s:13s} n={row['items']:4d} behaviour={row['behaviour']['value']:.3f} probe={r['probe']['value']:.3f} [{r['probe']['low']:.3f},{r['probe']['high']:.3f}] behaviour_given_probe={r['behaviour_given_probe']['value']:.3f} gap={r['gap']['value']:+.3f}")
        for label,row in out["differences"].items():
            r = row[view]
            b = r.get("e2b_behaviour_given_probe",r.get("behaviour_given_probe"))
            g = r.get("e2c_gap",r.get("gap"))
            print(f"  {label}: e2b {b['value']:+.3f} [{b['low']:+.3f},{b['high']:+.3f}] e2c {g['value']:+.3f} [{g['low']:+.3f},{g['high']:+.3f}]")
