import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from nwf import config
from nwf import maxsim
from nwf import models
from nwf import tokens
from nwf.paths import CAPTIONS
from nwf.paths import RESULTS
from nwf.paths import SCRATCH
from nwf.paths import STIMULI

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def capture(model):
    """Hook the input of every decoder layer and of the final norm and return the store they fill"""
    text = model.model.language_model
    store = {"on":False,"positions":None,"states":[]}

    def hook(module,args):
        """Keep the asked positions of the stream entering this module"""
        if store["on"]:
            store["states"].append(args[0][0,store["positions"].to(args[0].device)].float().to(DEVICE))

    for layer in text.layers:
        layer.register_forward_pre_hook(hook)
    text.norm.register_forward_pre_hook(hook)
    return store


def forward(model,store,enc,positions):
    """Run one forward pass and return the last letter logits row and the hooked states at the given positions"""
    store["positions"] = torch.tensor(positions)
    store["states"] = []
    store["on"] = True
    with torch.no_grad():
        out = model(**enc.to(DEVICE),logits_to_keep=1)
    store["on"] = False
    return out.logits[0,-1].float(),store["states"]


def encode(processor,images,prompt,prefill):
    """Build the chat input for some images and a question with the answer start appended"""
    content = [{"type":"image"} for _ in images]+[{"type":"text","text":prompt}]
    text = processor.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)+prefill
    return processor(text=[text],images=images,return_tensors="pt")


def letter_ids(tokenizer):
    """Return the token ids of the four answer letters right after the answer start"""
    start = tokenizer(config.PREFILL,add_special_tokens=False)["input_ids"]
    ids = []
    for letter in config.LETTERS:
        full = tokenizer(config.PREFILL+letter,add_special_tokens=False)["input_ids"]
        # each letter must add exactly one token after the unchanged answer start
        assert full[:-1]==start,f"letter {letter} changes the answer start tokens"
        ids.append(full[-1])
    return ids


def locate(enc,row,family,tokenizer,images):
    """Return the token positions of the reference shape the four targets and the prompt after the images"""
    names = config.FAMILIES[family]
    start,end = tokenizer.convert_tokens_to_ids([names["start"],names["end"]])
    ids = enc["input_ids"][0]
    ref_box = row["ref_boxes"][row["ref_index"]]
    if family=="qwen":
        grid = enc["image_grid_thw"]
        regions = [tokens.qwen_tokens(ids,grid[0],images[0].size,ref_box,0,start,end)]
        regions += [tokens.qwen_tokens(ids,grid[1],images[1].size,box,1,start,end) for box in row["tgt_boxes"]]
    else:
        # the token slots are the reference picture then its crops then the target picture then its crops
        size,most,ratio = names["pan_and_scan"]
        ref_crops = tokens.gemma_crops(*images[0].size,size,most,ratio)
        tgt_crops = tokens.gemma_crops(*images[1].size,size,most,ratio)
        regions = [tokens.gemma_tokens(ids,ref_crops,ref_box,1,start)]
        regions += [tokens.gemma_tokens(ids,tgt_crops,box,2+len(ref_crops),start) for box in row["tgt_boxes"]]
    assert all(regions),"a shape box maps to no tokens"
    return regions,tokens.after_images(ids,end)


def load_rows(name,limit):
    """Read the items of one set and keep the first ones when a limit is given"""
    rows = [json.loads(line) for line in (STIMULI/name/"items.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows[:limit] if limit else rows


def save(path,arrays,meta):
    """Write arrays and a done marker so a stopped run can resume without redoing finished parts"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp,**arrays)
    os.replace(tmp,path)
    path.with_suffix(".done.json").write_text(json.dumps(meta),encoding="utf-8")


def top_directions(gram,total,count,k):
    """Return the top principal directions and their variances from summed outer products"""
    mean = total/count
    cov = (gram-count*torch.outer(mean,mean))/count
    values,vectors = torch.linalg.eigh(cov)
    return vectors[:,-k:].flip(1).float(),values[-k:].flip(0).float()


def accumulate(sums,x):
    """Add the outer products and sums of a batch of stream states one stream at a time to save memory"""
    if sums["shift"] is None:
        # shifting by the first batch mean stops a large shared offset from swamping the variance
        sums["shift"] = x.mean(1,keepdim=True)
        sums["gram"] = torch.zeros(x.shape[0],x.shape[2],x.shape[2],dtype=torch.float64,device=DEVICE)
        sums["total"] = torch.zeros(x.shape[0],x.shape[2],dtype=torch.float64,device=DEVICE)
    x = x-sums["shift"]
    for s in range(x.shape[0]):
        sums["gram"][s] += (x[s].T@x[s]).double()
    sums["total"] += x.sum(1).double()
    sums["count"] += x.shape[1]


def fit_bases(model,processor,store,name,mode,family,n_captions,n_items):
    """Fit the vocabulary caption and image token directions plus the massive dimensions and save them"""
    out = RESULTS/f"{name}_{mode}"/"bases.npz"
    if out.with_suffix(".done.json").exists():
        return
    started = time.time()
    k = max(config.KS)
    weight = model.get_output_embeddings().weight
    d = weight.shape[1]
    assert d>=k,"the model is narrower than the largest subspace"
    gram = torch.zeros(d,d,dtype=torch.float64,device=DEVICE)
    total = torch.zeros(d,dtype=torch.float64,device=DEVICE)
    for chunk in weight.detach().split(16384):
        chunk = chunk.to(DEVICE).double()
        gram += chunk.T@chunk
        total += chunk.sum(0)
    vocab,vocab_values = top_directions(gram,total,weight.shape[0],k)
    del gram

    tokenizer = processor.tokenizer
    captions = CAPTIONS.read_text(encoding="utf-8").splitlines()[:n_captions]
    streams = len(model.model.language_model.layers)+1
    sums = {"shift":None,"count":0}
    for caption in captions:
        ids = tokenizer(caption,return_tensors="pt")["input_ids"]
        assert ids.shape[1]>1,"a caption must have more than one token"
        # the first token is left out because language models park huge activations there
        store["positions"] = torch.arange(1,ids.shape[1])
        store["states"] = []
        store["on"] = True
        with torch.no_grad():
            model(input_ids=ids.to(DEVICE),logits_to_keep=1)
        store["on"] = False
        accumulate(sums,torch.stack(store["states"]))
    caption_fit = [top_directions(sums["gram"][s],sums["total"][s],sums["count"],k) for s in range(streams)]
    caption_tokens = sums["count"]
    del sums

    image_id = tokenizer.convert_tokens_to_ids(config.FAMILIES[family]["image"])
    sums = {"shift":None,"count":0}
    absolute = torch.zeros(streams,d,dtype=torch.float64,device=DEVICE)
    # share of the image token energy lying in the top caption directions which sizes the random control of the removal check
    energy = torch.zeros(streams,dtype=torch.float64,device=DEVICE)
    caught = torch.zeros(streams,len(config.REMOVAL_KS),dtype=torch.float64,device=DEVICE)
    for set_name in config.SETS:
        for row in load_rows(set_name,n_items):
            images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
            enc = encode(processor,images,config.PROMPT,config.PREFILL)
            _,states = forward(model,store,enc,tokens.image_positions(enc["input_ids"][0],image_id))
            x = torch.stack(states)
            absolute += x.abs().sum(1).double()
            energy += (x**2).sum((1,2)).double()
            for s in range(streams):
                for j,k_removed in enumerate(config.REMOVAL_KS):
                    caught[s,j] += float(((x[s]@caption_fit[s][0][:,:k_removed])**2).sum())
            accumulate(sums,x)
    image_fit = [top_directions(sums["gram"][s],sums["total"][s],sums["count"],k) for s in range(streams)]
    count = sums["count"]
    del sums
    absolute = (absolute/count).float().cpu()
    # a massive dimension is far larger than the median one and only the largest few are dropped
    massive = torch.zeros(streams,d,dtype=torch.bool)
    for s in range(streams):
        order = absolute[s].argsort(descending=True)[:config.MASSIVE_MAX]
        order = order[absolute[s][order]>config.MASSIVE_RATIO*absolute[s].median()]
        massive[s,order] = True

    arrays = {
        "vocab":vocab.cpu().numpy(),"vocab_values":vocab_values.cpu().numpy(),
        "caption":torch.stack([f[0] for f in caption_fit]).cpu().numpy(),"caption_values":torch.stack([f[1] for f in caption_fit]).cpu().numpy(),
        "image":torch.stack([f[0] for f in image_fit]).cpu().numpy(),"image_values":torch.stack([f[1] for f in image_fit]).cpu().numpy(),
        "massive":massive.numpy(),"absolute":absolute.cpu().numpy(),
        "caught_share":(caught/energy[:,None]).float().cpu().numpy(),
    }
    meta = {"model":name,"mode":mode,"captions":len(captions),"caption_tokens":caption_tokens,"image_tokens":count,"items_per_set":n_items,"seconds":round(time.time()-started,1)}
    save(out,arrays,meta)
    print(f"{name} {mode}: bases from {len(captions)} captions and {count} image tokens in {meta['seconds']} s")


def load_bases(name,mode,d):
    """Load the fitted directions onto the device and draw the seeded random subspaces"""
    saved = np.load(RESULTS/f"{name}_{mode}"/"bases.npz")
    bases = {key:torch.from_numpy(saved[key]).to(DEVICE) for key in ("vocab","caption","image","massive")}
    k = max(config.KS)
    assert d>=k and bases["vocab"].shape[1]==k,"the saved bases do not match the subspace sizes"
    randoms = []
    for i in range(config.RANDOM_SUBSPACES):
        gauss = np.random.default_rng([config.RANDOM_SEED,i]).standard_normal((d,k))
        randoms.append(np.linalg.qr(gauss)[0])
    # the random bases sit side by side so all their coordinates come from one large product
    bases["random"] = torch.from_numpy(np.concatenate(randoms,axis=1)).float().to(DEVICE)
    return bases


def score_items(model,processor,store,name,mode,family,limit,vision,sets=None,stop_at=None):
    """Score every item chunk by chunk and save the letters probe scores pooled vectors and norms"""
    tokenizer = processor.tokenizer
    letters = letter_ids(tokenizer)
    d = model.get_output_embeddings().weight.shape[1]
    bases = load_bases(name,mode,d)
    image_id = tokenizer.convert_tokens_to_ids(config.FAMILIES[family]["image"])
    deep = []
    if vision and family=="qwen":
        for merger in model.model.visual.deepstack_merger_list:
            merger.register_forward_hook(lambda module,args,output:deep.append(output.float().cpu()))
    for set_name in sets or config.SETS:
        rows = load_rows(set_name,limit)
        for first in range(0,len(rows),config.CHUNK):
            out = RESULTS/f"{name}_{mode}"/set_name/f"{first:04d}.npz"
            if out.with_suffix(".done.json").exists():
                continue
            # no new chunk starts after the deadline so the session can still save what is finished
            if stop_at and time.time()>stop_at:
                print(f"{name} {mode}: deadline reached before {set_name} {first}")
                return
            started = time.time()
            keep = {"logits":[],"top":[],"scores":[],"random":[],"pooled":[],"norms":[],"sizes":[]}
            for row in rows[first:first+config.CHUNK]:
                images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
                enc = encode(processor,images,config.PROMPT,config.PREFILL)
                regions,prompt = locate(enc,row,family,tokenizer,images)
                wanted = sum(regions,[])+prompt
                cuts = np.cumsum([len(r) for r in regions]+[len(prompt)])
                if vision:
                    wanted += tokens.image_positions(enc["input_ids"][0],image_id)
                deep.clear()
                logits,states = forward(model,store,enc,wanted)
                if vision:
                    # the precision check compares the image tokens entering the language model across the two modes
                    folder = SCRATCH/"vision"/f"{name}_{mode}"
                    folder.mkdir(parents=True,exist_ok=True)
                    torch.save({"stream0":states[0][cuts[-1]:].cpu(),"deepstack":list(deep)},folder/f"{row['id']}.pt")
                scores,randoms,pooled,norms = [],[],[],[]
                for s,x in enumerate(states):
                    parts = torch.tensor_split(x[:cuts[-1]],cuts[:-1].tolist())
                    ref,targets,text = parts[0],list(parts[1:5]),parts[5]
                    pooled.append(torch.stack([p.mean(0) for p in parts[:5]]))
                    norms.append(torch.stack([torch.cat(parts[:5]).norm(dim=-1).mean(),text.norm(dim=-1).mean()]))
                    # the second view drops the massive dimensions before anything is compared
                    kept = (~bases["massive"][s]).float()
                    views = [maxsim.stream_scores(ref,targets,text,bases,s),maxsim.stream_scores(ref*kept,[t*kept for t in targets],text*kept,bases,s)]
                    scores.append(torch.stack([v[0] for v in views]))
                    randoms.append(torch.stack([v[1] for v in views]))
                keep["logits"].append(logits[letters].cpu())
                keep["top"].append(int(logits.argmax()))
                keep["scores"].append(torch.stack(scores,1).cpu())
                keep["random"].append(torch.stack(randoms,1).cpu())
                keep["pooled"].append(torch.stack(pooled).half().cpu())
                keep["norms"].append(torch.stack(norms).cpu())
                keep["sizes"].append([len(r) for r in regions]+[len(prompt)])
            ids = [row["id"] for row in rows[first:first+config.CHUNK]]
            arrays = {"ids":np.array(ids),"top":np.array(keep["top"]),"sizes":np.array(keep["sizes"])}
            arrays.update({key:torch.stack(keep[key]).numpy() for key in ("logits","scores","random","pooled","norms")})
            meta = {"model":name,"mode":mode,"set":set_name,"first":first,"items":len(ids),"letter_ids":letters,"views":config.VIEWS,"seconds":round(time.time()-started,1)}
            save(out,arrays,meta)
        print(f"{name} {mode}: {set_name} scored")


def generate(model,processor,images,prompt,prefill,new):
    """Greedily continue a prompt and return only the new text"""
    enc = encode(processor,images,prompt,prefill).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc,max_new_tokens=new,do_sample=False)
    return processor.tokenizer.decode(out[0,enc["input_ids"].shape[1]:],skip_special_tokens=True)


def smoke_checks(model,processor,name,mode,limit):
    """Save the authors style generated answers and the shape naming answers for the smoke run"""
    answers = {}
    for set_name in config.SETS:
        for row in load_rows(set_name,limit)[:config.GENERATE_ITEMS]:
            images = [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]
            answers[row["id"]] = generate(model,processor,images,config.PROMPT,config.PREFILL,config.ANSWER_TOKENS)
    naming = {}
    for set_name in ("named_fill","named_line","named_cx"):
        for path in sorted((STIMULI/set_name/"shapes").glob("*.png")):
            naming[f"{set_name}/{path.stem}"] = generate(model,processor,[Image.open(path).convert("RGB")],config.NAMING_PROMPT,"",config.NAMING_TOKENS)
    out = RESULTS/f"{name}_{mode}"/"smoke_checks.json"
    out.write_text(json.dumps({"answers":answers,"naming":naming},indent=1),encoding="utf-8")
    print(f"{name} {mode}: {len(answers)} generated answers and {len(naming)} shape names saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run one model over the stimulus sets")
    parser.add_argument("--model",required=True)
    parser.add_argument("--mode",choices=["full","fast"],required=True)
    parser.add_argument("--smoke",action="store_true",help="only the first items of each set plus the extra smoke checks")
    parser.add_argument("--sets",help="comma separated sets to score instead of all of them")
    parser.add_argument("--stop-at",type=float,help="unix time after which no new chunk starts")
    args = parser.parse_args()
    family = config.MODELS[args.model]["family"]
    model,processor = models.load(args.model,args.mode)
    store = capture(model)
    limit = config.SMOKE_ITEMS if args.smoke else None
    n_captions = 200 if args.smoke else config.CAPTIONS
    n_items = 2 if args.smoke else config.BASIS_ITEMS
    fit_bases(model,processor,store,args.model,args.mode,family,n_captions,n_items)
    score_items(model,processor,store,args.model,args.mode,family,limit,args.smoke,args.sets.split(",") if args.sets else None,args.stop_at)
    # the authors style answers and the naming check only need the fp32 run
    if args.smoke and args.mode=="full":
        smoke_checks(model,processor,args.model,args.mode,limit)
