import json
import math

import pytest
import torch
from PIL import Image
from PIL import ImageDraw
from transformers import AutoProcessor

from helpers import authors_functions
from nwf import config
from nwf import tokens
from nwf.paths import STIMULI
from nwf.run_items import encode
from nwf.run_items import locate

BOX = [[210,160],[330,280]]


def processor(family):
    """Load the pinned processor of a model family the way the runs do"""
    spec = config.MODELS[family]
    extra = {"do_pan_and_scan":True} if family=="gemma" else {}
    return AutoProcessor.from_pretrained(spec["repo"],revision=spec["revision"],use_fast=config.FAMILIES[family]["use_fast"],**extra)


def sample_rows():
    """Return a few real items from every set including authors items whose boxes cross the picture edge"""
    rows = []
    for name in config.SETS:
        rows += [json.loads(line) for line in (STIMULI/name/"items.jsonl").read_text(encoding="utf-8").splitlines()[:3]]
    rep = [json.loads(line) for line in (STIMULI/"rep_named"/"items.jsonl").read_text(encoding="utf-8").splitlines()]
    rows += [r for r in rep if any(x0<0 or y0<0 or x1>800 or y1>600 for (x0,y0),(x1,y1) in r["ref_boxes"]+r["tgt_boxes"])][:5]
    return rows


def pictures(row):
    """Open the two pictures of an item"""
    return [Image.open(STIMULI/row[key]).convert("RGB") for key in ("ref_image","tgt_image")]


def test_qwen_tokens_match_authors():
    proc = processor("qwen")
    theirs = authors_functions("rep_probe_qwen.py",("find_qwen3vl_image_tokens","get_processed_resolution","scale_bbox"),{"torch":torch})
    for row in sample_rows():
        images = pictures(row)
        enc = encode(proc,images,config.PROMPT,config.PREFILL)
        regions,_ = locate(enc,row,"qwen",proc.tokenizer,images)
        expected = []
        for image_index,box in [(0,row["ref_boxes"][row["ref_index"]])]+[(1,b) for b in row["tgt_boxes"]]:
            h,w = theirs["get_processed_resolution"](enc,image_index)
            scaled = theirs["scale_bbox"](box,images[image_index],h,w)
            expected.append(theirs["find_qwen3vl_image_tokens"](enc["input_ids"][0],(h,w),scaled,image_index=image_index))
        assert regions==expected


def test_gemma_tokens_match_authors():
    proc = processor("gemma")
    theirs = authors_functions("rep_probe_gemma.py",("compute_pan_and_scan_crops","find_tokens_in_crops"),{"math":math,"torch":torch})
    for row in sample_rows():
        images = pictures(row)
        enc = encode(proc,images,config.PROMPT,config.PREFILL)
        regions,_ = locate(enc,row,"gemma",proc.tokenizer,images)
        crops_ref = theirs["compute_pan_and_scan_crops"](*images[0].size)
        crops_tgt = theirs["compute_pan_and_scan_crops"](*images[1].size)
        to_tuple = lambda b:(b[0][0],b[0][1],b[1][0],b[1][1])
        expected = [theirs["find_tokens_in_crops"](enc["input_ids"],to_tuple(row["ref_boxes"][row["ref_index"]]),crops_ref,1)]
        expected += [theirs["find_tokens_in_crops"](enc["input_ids"],to_tuple(b),crops_tgt,1+len(crops_ref)+1) for b in row["tgt_boxes"]]
        assert regions==expected


@pytest.mark.parametrize("family",["qwen","gemma"])
def test_positions_are_image_tokens_and_prompt_follows(family):
    proc = processor(family)
    image_id = proc.tokenizer.convert_tokens_to_ids(config.FAMILIES[family]["image"])
    for row in sample_rows()[::4]:
        images = pictures(row)
        enc = encode(proc,images,config.PROMPT,config.PREFILL)
        ids = enc["input_ids"][0]
        regions,prompt = locate(enc,row,family,proc.tokenizer,images)
        for region in regions:
            assert len(region)==len(set(region))>0
            assert all(int(ids[p])==image_id for p in region)
        assert prompt[-1]==len(ids)-1
        assert all(int(ids[p])!=image_id for p in prompt)
        assert proc.tokenizer.decode(ids[prompt]).endswith(config.PREFILL)


def test_qwen_tokens_cover_a_drawn_box():
    proc = processor("qwen")
    image = Image.new("RGB",(800,600),"black")
    ImageDraw.Draw(image).rectangle([BOX[0][0],BOX[0][1],BOX[1][0]-1,BOX[1][1]-1],fill="white")
    enc = encode(proc,[image,image],config.PROMPT,config.PREFILL)
    ids = enc["input_ids"][0]
    start,end = proc.tokenizer.convert_tokens_to_ids(["<|vision_start|>","<|vision_end|>"])
    mapped = tokens.qwen_tokens(ids,enc["image_grid_thw"][0],image.size,BOX,0,start,end)
    # each language model token is four neighbouring patches stored one after another
    first = [i for i,t in enumerate(ids.tolist()) if t==start][0]+1
    per_token = enc["pixel_values"][:len(enc["pixel_values"])//2].reshape(-1,4,enc["pixel_values"].shape[1])
    bright = [first+i for i in range(len(per_token)) if per_token[i].max()>0]
    assert mapped==bright


def test_gemma_crops_match_the_processor():
    proc = processor("gemma")
    row = sample_rows()[0]
    images = pictures(row)
    enc = encode(proc,images,config.PROMPT,config.PREFILL)
    size,most,ratio = config.FAMILIES["gemma"]["pan_and_scan"]
    crops = tokens.gemma_crops(*images[0].size,size,most,ratio)
    assert len(enc["pixel_values"])==2*(1+len(crops))
    plain = AutoProcessor.from_pretrained(config.MODELS["gemma"]["repo"],revision=config.MODELS["gemma"]["revision"],use_fast=False,do_pan_and_scan=False)
    for slot,box in enumerate([None]+crops):
        part = images[0] if box is None else images[0].crop(box)
        alone = plain(text=["<start_of_image>"],images=[part],return_tensors="pt")["pixel_values"][0]
        assert torch.equal(alone,enc["pixel_values"][slot])


def test_gemma_tokens_lie_on_a_drawn_box():
    proc = processor("gemma")
    image = Image.new("RGB",(800,600),"black")
    ImageDraw.Draw(image).rectangle([BOX[0][0],BOX[0][1],BOX[1][0]-1,BOX[1][1]-1],fill="white")
    enc = encode(proc,[image,image],config.PROMPT,config.PREFILL)
    ids = enc["input_ids"][0]
    start = proc.tokenizer.convert_tokens_to_ids("<start_of_image>")
    size,most,ratio = config.FAMILIES["gemma"]["pan_and_scan"]
    crops = tokens.gemma_crops(800,600,size,most,ratio)
    mapped = tokens.gemma_tokens(ids,crops,BOX,1,start)
    starts = [i for i,t in enumerate(ids.tolist()) if t==start]
    # each language model token is the average of one 56 pixel block of a 896 pixel crop
    bright = []
    for c in range(len(crops)):
        blocks = enc["pixel_values"][1+c].amax(0).reshape(16,56,16,56).amax((1,3))
        bright += [starts[1+c]+1+int(i) for i in torch.nonzero(blocks.flatten()>0).flatten()]
    # the box spans token columns 8 to 13 and rows 4 to 7 of the left crop
    assert len(mapped)==5*3
    assert set(mapped)<=set(bright)
    # the authors leave out the blocks the box only partly enters on its right and bottom side
    missing = {divmod(p-starts[1]-1,16) for p in set(bright)-set(mapped)}
    assert missing and all(y==7 or x==13 for y,x in missing)
