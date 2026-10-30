import argparse
import json

import numpy as np
import torch

from nwf.paths import RESULTS
from nwf.paths import SCRATCH


def chunks(name,mode):
    """Load every saved chunk of a smoke run into one dictionary of arrays"""
    files = sorted((RESULTS/f"{name}_{mode}").glob("*/*.npz"))
    loaded = [np.load(f) for f in files if f.name!="bases.npz"]
    return {key:np.concatenate([c[key] for c in loaded]) for key in ("ids","logits","scores","pooled","norms")}


def compare(name):
    """Judge whether the fast mode of a model may replace full fp32 and write the verdict"""
    full,fast = chunks(name,"full"),chunks(name,"fast")
    assert list(full["ids"])==list(fast["ids"]),"the two runs scored different items"
    finite = all(np.isfinite(run[key].astype(np.float64)).all() for run in (full,fast) for key in ("logits","scores","pooled","norms"))

    # image tokens entering the language model averaged over each item and for qwen also the extra vision features
    entering = []
    for item in full["ids"]:
        a = torch.load(SCRATCH/"vision"/f"{name}_full"/f"{item}.pt")
        b = torch.load(SCRATCH/"vision"/f"{name}_fast"/f"{item}.pt")
        pairs = [(a["stream0"],b["stream0"])]+list(zip(a["deepstack"],b["deepstack"]))
        finite = finite and all(torch.isfinite(x).all() for pair in pairs for x in pair)
        entering.append(min(float(torch.nn.functional.cosine_similarity(x,y,dim=-1).mean()) for x,y in pairs))

    # an answer flip is allowed only where full fp32 was torn between its top two letters
    probs = torch.softmax(torch.from_numpy(full["logits"]).double(),-1).numpy()
    top_two = np.sort(probs,-1)[:,-2:]
    near_tie = (top_two[:,1]-top_two[:,0])<0.05
    flips = full["logits"].argmax(-1)!=fast["logits"].argmax(-1)
    same = 1-flips.mean()

    pooled_cos = []
    for s in range(full["pooled"].shape[1]):
        a = torch.from_numpy(full["pooled"][:,s].astype(np.float32))
        b = torch.from_numpy(fast["pooled"][:,s].astype(np.float32))
        pooled_cos.append(float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean()))
    probe_same = (full["scores"][:,0,:,0].argmax(-1)==fast["scores"][:,0,:,0].argmax(-1)).mean(0)

    passed = bool(finite and min(entering)>=0.999 and same>=0.98 and near_tie[flips].all())
    verdict = {
        "model":name,"passed":passed,"items":len(full["ids"]),"no_nan_or_inf":bool(finite),
        "entering_cos_min":min(entering),"entering_cos_mean":float(np.mean(entering)),
        "same_letter":float(same),"flips":int(flips.sum()),"flips_all_near_ties":bool(near_tie[flips].all()),
        "pooled_region_cos_per_stream":pooled_cos,"probe_same_answer_per_stream":probe_same.tolist(),
    }
    (RESULTS/f"precision_{name}.json").write_text(json.dumps(verdict,indent=1),encoding="utf-8")
    print(f"{name}: precision check {'passed' if passed else 'failed'}, same letter {same:.3f}, entering cos min {min(entering):.5f}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare the full and fast smoke runs of one model")
    parser.add_argument("--model",required=True)
    compare(parser.parse_args().model)
