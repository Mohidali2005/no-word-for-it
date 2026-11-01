import argparse
import json

import numpy as np
import statsmodels.api as sm

from nwf.config import SETS
from nwf.paths import ANALYSIS
from nwf.paths import RESULTS
from nwf.paths import STIMULI

RESAMPLES = 10000
CORE = ["named_fill","named_line","unnamed_fill","unnamed_line"]
# pairs of sets where the first is expected to beat the second
DIFFERENCES = {
    "e1a_main_named_fill_minus_unnamed_line":("named_fill","unnamed_line"),
    "e1a_replication_rep_named_minus_rep_squig30":("rep_named","rep_squig30"),
    "e1b_filled_named_minus_unnamed":("named_fill","unnamed_fill"),
    "e1b_line_named_minus_unnamed":("named_line","unnamed_line"),
}


def load(run,name,set_name):
    """Return one entry per scored item with its shape cluster and whether the model got it right"""
    items = {}
    for line in open(STIMULI/set_name/"items.jsonl",encoding="utf-8"):
        row = json.loads(line)
        items[row["id"]] = row
    files = sorted((RESULTS/run/name/set_name).glob("*.npz"))
    loaded = [np.load(f) for f in files]
    ids = np.concatenate([c["ids"] for c in loaded])
    pred = np.concatenate([c["logits"] for c in loaded]).argmax(-1)
    rows = [items[i] for i in ids]
    answer = np.array(["ABCD".index(r["answer"]) for r in rows])
    # the released sets are all calibration items so they are used whole as a replication check
    if SETS[set_name]["kind"]=="rep":
        test = np.ones(len(rows),dtype=bool)
    else:
        test = np.array([not r["calibration"] for r in rows])
    return {
        "pred":pred[test],
        "answer":answer[test],
        "correct":(pred==answer)[test],
        "cluster":np.array([r["shapes"][r["ref_index"]] for r in rows])[test],
        "named":np.array([r["named"] for r in rows])[test],
        "filled":np.array([r["filled"] for r in rows])[test],
        "logcx":np.array([np.log(r["complexity"][r["ref_index"]]) for r in rows])[test],
        "lookalike":np.array([r["lookalike"] for r in rows])[test],
    }


def counts(d):
    """Count the correct and all items per shape cluster and answer letter"""
    clusters = sorted(set(d["cluster"]))
    k = np.zeros((len(clusters),4))
    n = np.zeros((len(clusters),4))
    for c,cluster in enumerate(clusters):
        for letter in range(4):
            here = (d["cluster"]==cluster)&(d["answer"]==letter)
            k[c,letter] = d["correct"][here].sum()
            n[c,letter] = here.sum()
    return k,n


def accuracies(k,n):
    """Plain accuracy and the mean of the four per letter accuracies from summed counts"""
    k,n = k.sum(-2),n.sum(-2)
    with np.errstate(invalid="ignore",divide="ignore"):
        balanced = np.nanmean(k/n,axis=-1)
    return k.sum(-1)/n.sum(-1),balanced


def resample(k,n,rng):
    """Bootstrap both accuracies by drawing whole shape clusters with replacement"""
    idx = rng.integers(0,len(k),(RESAMPLES,len(k)))
    return accuracies(k[idx],n[idx])


def interval(point,samples):
    """Point value with the 2.5 and 97.5 percent bootstrap limits"""
    low,high = np.percentile(samples,[2.5,97.5])
    return {"value":float(point),"low":float(low),"high":float(high)}


def run_model(run,name):
    """Compute the set table the paired differences and the regression of experiment one for one model"""
    rng = np.random.default_rng(0)
    data = {s:load(run,name,s) for s in SETS if (RESULTS/run/name/s).exists()}
    table = {}
    boots = {}
    for s,d in data.items():
        k,n = counts(d)
        plain,balanced = accuracies(k,n)
        boots[s] = (plain,balanced)+resample(k,n,rng)
        table[s] = {
            "items":len(d["pred"]),
            "clusters":len(k),
            "accuracy":interval(plain,boots[s][2]),
            "letter_balanced_accuracy":interval(balanced,boots[s][3]),
            "predicted_letter_share":(np.bincount(d["pred"],minlength=4)/len(d["pred"])).round(3).tolist(),
        }
    diffs = {}
    for label,(a,b) in DIFFERENCES.items():
        diffs[label] = {
            "accuracy":interval(boots[a][0]-boots[b][0],boots[a][2]-boots[b][2]),
            "letter_balanced_accuracy":interval(boots[a][1]-boots[b][1],boots[a][3]-boots[b][3]),
        }
    # the two within filled and within line differences averaged into one number
    f,l = diffs["e1b_filled_named_minus_unnamed"],diffs["e1b_line_named_minus_unnamed"]
    diffs["e1b_main_mean_of_filled_and_line"] = {
        "accuracy":interval((boots["named_fill"][0]-boots["unnamed_fill"][0]+boots["named_line"][0]-boots["unnamed_line"][0])/2,
            (boots["named_fill"][2]-boots["unnamed_fill"][2]+boots["named_line"][2]-boots["unnamed_line"][2])/2),
    }

    # logistic regression on the four core sets with errors clustered by shape identity
    core = {key:np.concatenate([data[s][key] for s in CORE]) for key in data["named_fill"]}
    X = sm.add_constant(np.column_stack([core["named"],core["filled"],core["logcx"],core["lookalike"]]).astype(float))
    groups = np.unique(core["cluster"],return_inverse=True)[1]
    fit = sm.Logit(core["correct"].astype(float),X).fit(disp=0,cov_type="cluster",cov_kwds={"groups":groups})
    low,high = fit.conf_int()[1]
    regression = {
        "terms":["const","named","filled","log_complexity","lookalike"],
        "coef":fit.params.round(4).tolist(),
        "se":fit.bse.round(4).tolist(),
        "named_coef_ci95":[float(low),float(high)],
        "named_p":float(fit.pvalues[1]),
        "items":int(len(groups)),
        "clusters":int(groups.max()+1),
    }
    return {"model":name,"sets":table,"differences":diffs,"regression":regression}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment one behaviour gap for one model run")
    parser.add_argument("--run",default="kaggle_run_01/results")
    parser.add_argument("--name",required=True,help="folder of the model run for example qwen_fast")
    args = parser.parse_args()
    out = run_model(args.run,args.name)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS/f"e1_{args.name}.json").write_text(json.dumps(out,indent=1),encoding="utf-8")
    for s,row in out["sets"].items():
        a = row["accuracy"]
        print(f"{s:13s} n={row['items']:4d} acc={a['value']:.3f} [{a['low']:.3f},{a['high']:.3f}] balanced={row['letter_balanced_accuracy']['value']:.3f} letters={row['predicted_letter_share']}")
    for label,row in out["differences"].items():
        a = row["accuracy"]
        print(f"{label}: {a['value']:+.3f} [{a['low']:+.3f},{a['high']:+.3f}]")
    r = out["regression"]
    print(f"named coef {r['coef'][1]:+.3f} ci {r['named_coef_ci95'][0]:+.3f},{r['named_coef_ci95'][1]:+.3f} p={r['named_p']:.3g}")
