import argparse
import time
from functools import partial

import numpy as np
import torch
from PIL import Image

from nwf import config
from nwf import models
from nwf import tokens
from nwf.paths import RESULTS
from nwf.paths import STIMULI
from nwf.run_items import DEVICE
from nwf.run_items import encode
from nwf.run_items import letter_ids
from nwf.run_items import load_rows
from nwf.run_items import save


def conditions(name,mode,d):
    """Return for every removal condition the directions to take out at each stream index"""
    saved = np.load(RESULTS/f"{name}_{mode}"/"bases.npz")
    caption,share = saved["caption"],saved["caught_share"]
    found = {"none":None}
    sizes = {}
    for j,k in enumerate(config.REMOVAL_KS):
        found[f"caption{k}"] = [torch.from_numpy(c[:,:k]).float() for c in caption]
        # a random subspace of size m holds m over d of any vector's energy on average so m matches the caption share
        sizes[k] = [int(min(d,max(k,round(float(share[s,j])*d)))) for s in range(len(caption))]
        for draw in range(config.REMOVAL_DRAWS):
            gauss = np.random.default_rng([config.RANDOM_SEED,1000+100*k+draw]).standard_normal((d,max(sizes[k])))
            q = np.linalg.qr(gauss)[0]
            found[f"random{k}_{draw}"] = [torch.from_numpy(q[:,:m]).float() for m in sizes[k]]
    return found,sizes


def cut(module,args,positions,directions):
    """Take some directions out of the image tokens of the stream entering a layer"""
    x = args[0].clone()
    where = positions.to(x.device)
    part = x[0,where].float()
    u = directions.to(part.device)
    x[0,where] = (part-(part@u)@u.T).to(x.dtype)
    return (x,)+args[1:]


def run(model,processor,name,mode,family,limit,stop_at):
    """Score the letters of the removal items with no removal and with every caption and random removal"""
    tokenizer = processor.tokenizer
    letters = letter_ids(tokenizer)
    image_id = tokenizer.convert_tokens_to_ids(config.FAMILIES[family]["image"])
    layers = model.model.language_model.layers
    found,sizes = conditions(name,mode,model.get_output_embeddings().weight.shape[1])
    first_layer = config.REMOVAL_FIRST_LAYER[family]
    start,stop = (0,limit) if limit else config.REMOVAL_ITEMS
    for set_name in config.REMOVAL_SETS:
        rows = load_rows(set_name,None)[start:stop]
        for first in range(0,len(rows),config.CHUNK):
            out = RESULTS/f"{name}_{mode}"/"removal"/set_name/f"{start+first:04d}.npz"
            if out.with_suffix(".done.json").exists():
                continue
            if stop_at and time.time()>stop_at:
                print(f"{name} {mode}: deadline reached before removal {set_name} {start+first}")
                return
            began = time.time()
            logits = []
            for row in rows[first:first+config.CHUNK]:
                images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
                enc = encode(processor,images,config.PROMPT,config.PREFILL).to(DEVICE)
                positions = torch.tensor(tokens.image_positions(enc["input_ids"][0],image_id))
                per_condition = []
                for directions in found.values():
                    handles = []
                    if directions is not None:
                        handles = [layer.register_forward_pre_hook(partial(cut,positions=positions,directions=directions[s])) for s,layer in enumerate(layers) if s>=first_layer]
                    with torch.no_grad():
                        per_condition.append(model(**enc,logits_to_keep=1).logits[0,-1,letters].float().cpu())
                    for handle in handles:
                        handle.remove()
                logits.append(torch.stack(per_condition))
            ids = [row["id"] for row in rows[first:first+config.CHUNK]]
            meta = {"model":name,"mode":mode,"set":set_name,"first":start+first,"items":len(ids),"conditions":list(found),"random_sizes":sizes,"first_layer":first_layer,"seconds":round(time.time()-began,1)}
            save(out,{"ids":np.array(ids),"logits":torch.stack(logits).numpy()},meta)
        print(f"{name} {mode}: removal on {set_name} scored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the removal check for one model")
    parser.add_argument("--model",required=True)
    parser.add_argument("--mode",choices=["full","fast"],required=True)
    parser.add_argument("--smoke",action="store_true",help="only the first few items of each removal set")
    parser.add_argument("--stop-at",type=float,help="unix time after which no new chunk starts")
    args = parser.parse_args()
    model,processor = models.load(args.model,args.mode)
    run(model,processor,args.model,args.mode,config.MODELS[args.model]["family"],5 if args.smoke else None,args.stop_at)
