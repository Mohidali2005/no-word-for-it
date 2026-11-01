import argparse
import json

import numpy as np
import statsmodels.api as sm

from nwf.analysis.behaviour import CORE
from nwf.analysis.behaviour import DIFFERENCES
from nwf.analysis.behaviour import RESAMPLES
from nwf.analysis.behaviour import interval
from nwf.analysis.probe import PROBE_VIEWS
from nwf.config import KS
from nwf.config import MODELS
from nwf.config import REMOVAL_FIRST_LAYER
from nwf.config import SETS
from nwf.config import VIEWS
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS
from nwf.paths import STIMULI

MAIN_K = 64
# line squiggles from simple to very tangled with the main thirty anchor set in its place
SWEEP = ["squig_n5","squig_n10","squig_n20","unnamed_line","squig_n50"]
COVARIATES = ["named","filled","logcx","lookalike"]


def load(run,name,set_name):
    """Return per item the probe hits with and without the loud and random directions plus margins behaviour and covariates"""
    items = {}
    for line in open(STIMULI/set_name/"items.jsonl",encoding="utf-8"):
        row = json.loads(line)
        items[row["id"]] = row
    keep = {"ids":[],"full":[],"loud":[],"random":[],"margin":[],"margin_loud":[],"correct":[]}
    rows = []
    for f in sorted((RESULTS/run/name/set_name).glob("*.npz")):
        chunk = np.load(f)
        here = [items[i] for i in chunk["ids"]]
        answer = np.array(["ABCD".index(r["answer"]) for r in here])
        scores = chunk["scores"]
        hits = scores.argmax(-1)==answer[:,None,None,None]
        keep["full"].append(hits[...,0])
        # the residual view of the image token pca is the probe after the top k loud directions are taken out
        keep["loud"].append(np.stack([hits[...,VIEWS.index(f"image{k}_res")] for k in KS],-1))
        # the random answers are averaged over the subspaces and only the outside part of each size k is kept
        keep["random"].append((chunk["random"]==answer[:,None,None,None,None,None]).mean(3)[...,:len(KS),1])
        for key,view in (("margin",0),("margin_loud",VIEWS.index(f"image{MAIN_K}_res"))):
            s = scores[...,view,:]
            right = np.take_along_axis(s,answer[:,None,None,None],-1)[...,0]
            # masking the right target so the best wrong one is left
            wrong = np.where(np.arange(4)==answer[:,None,None,None],-np.inf,s).max(-1)
            keep[key].append(right-wrong)
        keep["correct"].append(chunk["logits"].argmax(-1)==answer)
        keep["ids"].append(chunk["ids"])
        rows += here
    out = {key:np.concatenate(value) for key,value in keep.items()}
    out["cluster"] = np.array([r["shapes"][r["ref_index"]] for r in rows])
    out["calibration"] = np.array([r["calibration"] for r in rows])
    # the released sets are all calibration items so they are used whole as a replication check
    out["test"] = np.ones(len(rows),dtype=bool) if SETS[set_name]["kind"]=="rep" else ~out["calibration"]
    out["named"] = np.array([r["named"] for r in rows])
    out["filled"] = np.array([r["filled"] for r in rows])
    out["logcx"] = np.array([np.log(r["complexity"][r["ref_index"]]) for r in rows])
    out["lookalike"] = np.array([r["lookalike"] for r in rows])
    return out


def bootstrap(values,cluster,rng):
    """Return the mean of per item values and its bootstrap draws over whole shape clusters"""
    groups = np.unique(cluster,return_inverse=True)[1]
    flat = values.reshape(len(values),-1).astype(np.float64)
    sums = np.zeros((groups.max()+1,flat.shape[1]))
    np.add.at(sums,groups,flat)
    counts = np.bincount(groups).astype(np.float64)
    draws = rng.integers(0,len(sums),(RESAMPLES,len(sums)))
    weights = np.zeros((RESAMPLES,len(sums)))
    np.add.at(weights,(np.arange(RESAMPLES)[:,None],draws),1)
    point = (sums.sum(0)/counts.sum()).reshape(values.shape[1:])
    boot = ((weights@sums)/(weights@counts)[:,None]).reshape(RESAMPLES,*values.shape[1:])
    return point,boot


def stack(d):
    """Put the full loud and random hits of the test items side by side as one array per item"""
    t = d["test"]
    return np.concatenate([d["full"][t][...,None],d["loud"][t],d["random"][t]],-1)


def boost_table(point,boot,layer,first):
    """Return accuracy with and without the loud and random directions and the excess gain at the probe layer and over streams"""
    k = KS.index(MAIN_K)
    full,loud,random = point[...,0],point[...,1:1+len(KS)],point[...,1+len(KS):]
    excess = boot[...,1+k]-boot[...,1+len(KS)+k]
    out = {}
    for v,view in enumerate(PROBE_VIEWS):
        out[view] = {
            "full_at_layer":round(float(full[v,layer[v]]),4),
            "loud_removed_at_layer":{f"k{kk}":round(float(loud[v,layer[v],j]),4) for j,kk in enumerate(KS)},
            "random_removed_at_layer":{f"k{kk}":round(float(random[v,layer[v],j]),4) for j,kk in enumerate(KS)},
            # the excess is the gain from taking out the loud directions beyond taking out random ones of the same size
            "excess_at_layer":interval(loud[v,layer[v],k]-random[v,layer[v],k],excess[:,v,layer[v]]),
            "excess_mean_over_streams":interval((loud[v,first:,k]-random[v,first:,k]).mean(),excess[:,v,first:].mean(-1)),
            "full_by_stream":full[v].round(4).tolist(),
            "loud_removed_by_stream":{f"k{kk}":loud[v,:,j].round(4).tolist() for j,kk in enumerate(KS)},
            "random_removed_by_stream":{f"k{kk}":random[v,:,j].round(4).tolist() for j,kk in enumerate(KS)},
        }
    return out


def difference(point_a,boot_a,point_b,boot_b,layer,first):
    """Return the excess gain of the second set minus the first at the probe layer and over streams"""
    k = KS.index(MAIN_K)
    ea,eb = point_a[...,1+k]-point_a[...,1+len(KS)+k],point_b[...,1+k]-point_b[...,1+len(KS)+k]
    ba,bb = boot_a[...,1+k]-boot_a[...,1+len(KS)+k],boot_b[...,1+k]-boot_b[...,1+len(KS)+k]
    return {view:{
        "at_layer":interval(eb[v,layer[v]]-ea[v,layer[v]],bb[:,v,layer[v]]-ba[:,v,layer[v]]),
        "mean_over_streams":interval((eb[v,first:]-ea[v,first:]).mean(),(bb[:,v,first:]-ba[:,v,first:]).mean(-1)),
    } for v,view in enumerate(PROBE_VIEWS)}


def logit(core,columns):
    """Fit correct on the given columns with errors clustered by shape and return every term with its interval"""
    X = sm.add_constant(np.column_stack([core[c] for c in columns]).astype(float))
    groups = np.unique(core["cluster"],return_inverse=True)[1]
    fit = sm.Logit(core["correct"].astype(float),X).fit(disp=0,cov_type="cluster",cov_kwds={"groups":groups})
    ci = fit.conf_int()
    return {c:{"coef":round(float(fit.params[j+1]),4),"low":round(float(ci[j+1][0]),4),"high":round(float(ci[j+1][1]),4),"p":float(fit.pvalues[j+1])} for j,c in enumerate(columns)}


def run_model(run,name,rng):
    """Compute the loudness gains per set and the margin regression for one model run"""
    data = {s:load(run,name,s) for s in SETS}
    first = REMOVAL_FIRST_LAYER[MODELS[name.rsplit("_",1)[0]]["family"]]
    # the probe layer is chosen exactly as in experiment two from the calibration items of all sets pooled
    pooled = np.concatenate([d["full"][d["calibration"]] for d in data.values()]).mean(0)
    layer = pooled.argmax(-1)
    stats = {s:bootstrap(stack(d),d["cluster"][d["test"]],rng) for s,d in data.items()}
    table = {s:boost_table(*stats[s],layer,first) for s in SETS}

    # every difference is the second set minus the first so a gain that favours unnamed shapes comes out positive
    pairs = list(DIFFERENCES.values())+[("squig_n5","squig_n50")]
    diffs = {f"{b}_minus_{a}":difference(*stats[a],*stats[b],layer,first) for a,b in pairs}

    # the margin is how far the right target beats the best wrong one at the probe layer of the raw view
    core = {}
    for key in ["correct","cluster","margin","margin_loud"]+COVARIATES:
        parts = []
        for s in CORE:
            d = data[s]
            value = d[key][d["test"]]
            parts.append(value[:,0,layer[0]] if key.startswith("margin") else value)
        core[key] = np.concatenate(parts)
    core["set"] = np.concatenate([[s]*int(data[s]["test"].sum()) for s in CORE])
    for key in ("margin","margin_loud"):
        core[key] = (core[key]-core[key].mean())/core[key].std()
    regression = {
        # without the margin this is the experiment one regression and must give the same named coefficient
        "without_margin":logit(core,COVARIATES),
        "with_margin":logit(core,COVARIATES+["margin"]),
        "with_loud_removed_margin":logit(core,COVARIATES+["margin_loud"]),
        # inside one set the name never changes so a margin slope here cannot come from names
        "margin_within_set":{s:logit({key:value[core["set"]==s] for key,value in core.items()},["margin"])["margin"] for s in CORE},
        "mean_margin_by_set":{s:round(float(data[s]["margin"][data[s]["test"]][:,0,layer[0]].mean()),5) for s in CORE},
        "items":int(len(core["correct"])),
    }
    probe_layer = {view:int(layer[v]) for v,view in enumerate(PROBE_VIEWS)}
    return {"model":name,"probe_layer":probe_layer,"first_stream":first,"main_k":MAIN_K,"sets":table,"differences":diffs,"regression":regression},data,layer,first


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exploratory check of whether loud image directions hide fine shape detail")
    parser.add_argument("--run",default="kaggle_run_01/results")
    parser.add_argument("--names",nargs="+",required=True,help="model run folders for example qwen_fast qwen_names_fast")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out = {"status":"exploratory","models":{}}
    loaded = {}
    for name in args.names:
        out["models"][name],data,layer,first = run_model(args.run,name,rng)
        loaded[name] = (data,layer,first)

    # the same items are scored by every model so the change in excess gain is bootstrapped item by item
    if len(args.names)==2:
        (a,(da,la,fa)),(b,(db,lb,fb)) = loaded.items()
        k = KS.index(MAIN_K)
        change = {}
        for s in SETS:
            ta,tb = da[s]["test"],db[s]["test"]
            assert (da[s]["ids"][ta]==db[s]["ids"][tb]).all(),"the two runs scored different items"
            # each model is read at its own probe layer in the raw view
            gain_a = da[s]["loud"][ta][:,0,la[0],k]-da[s]["random"][ta][:,0,la[0],k]
            gain_b = db[s]["loud"][tb][:,0,lb[0],k]-db[s]["random"][tb][:,0,lb[0],k]
            point,boot = bootstrap((gain_b-gain_a)[:,None],da[s]["cluster"][ta],rng)
            change[s] = interval(point[0],boot[:,0])
        out["change_in_excess_raw_view"] = {"second_minus_first":f"{b} minus {a}","sets":change}

    ANALYSIS.mkdir(exist_ok=True)
    path = ANALYSIS/f"loudness_{'_vs_'.join(args.names)}.json"
    path.write_text(json.dumps(out,indent=1),encoding="utf-8")
    for name,m in out["models"].items():
        print(f"{name}: probe layer {m['probe_layer']['raw']} excess gain from removing the top {MAIN_K} image directions beyond random ones")
        for s,row in m["sets"].items():
            r = row["raw"]
            e,em = r["excess_at_layer"],r["excess_mean_over_streams"]
            print(f"  {s:13s} full={r['full_at_layer']:.3f} loud_removed={r['loud_removed_at_layer'][f'k{MAIN_K}']:.3f} random_removed={r['random_removed_at_layer'][f'k{MAIN_K}']:.3f} excess={e['value']:+.3f} [{e['low']:+.3f},{e['high']:+.3f}] over_streams={em['value']:+.3f} [{em['low']:+.3f},{em['high']:+.3f}]")
        for label,row in m["differences"].items():
            e = row["raw"]["at_layer"]
            print(f"  {label}: {e['value']:+.3f} [{e['low']:+.3f},{e['high']:+.3f}]")
        for fit in ("without_margin","with_margin","with_loud_removed_margin"):
            r = m["regression"][fit]
            extra = "".join(f" {c}={r[c]['coef']:+.3f} [{r[c]['low']:+.3f},{r[c]['high']:+.3f}]" for c in r if c.startswith("margin"))
            print(f"  {fit}: named={r['named']['coef']:+.3f} [{r['named']['low']:+.3f},{r['named']['high']:+.3f}]{extra}")
        for s,r in m["regression"]["margin_within_set"].items():
            print(f"  margin slope within {s}: {r['coef']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] mean margin {m['regression']['mean_margin_by_set'][s]:+.5f}")
    if "change_in_excess_raw_view" in out:
        print(out["change_in_excess_raw_view"]["second_minus_first"])
        for s,e in out["change_in_excess_raw_view"]["sets"].items():
            print(f"  {s:13s} {e['value']:+.3f} [{e['low']:+.3f},{e['high']:+.3f}]")
