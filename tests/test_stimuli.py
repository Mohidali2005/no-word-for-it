import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from nwf import config
from nwf import shapes
from nwf import stimuli
from nwf.paths import AUTHORS_SHAPES


def authors_save_image():
    """Load the authors drawing functions straight from their script without running the rest of it"""
    source = (AUTHORS_SHAPES/"generate_dataset.py").read_text(encoding="utf-8")
    keep = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ("change_colors","save_image")]
    namespace = {"np":np,"plt":plt,"Image":Image}
    exec(compile(ast.Module(body=keep,type_ignores=[]),"generate_dataset","exec"),namespace)
    return namespace["save_image"]


@pytest.mark.parametrize("folder,seed",[("basic_shapes",1),("basic_shapes",2),("squiggles_30",3),("squiggles_30",4)])
def test_render_matches_authors(tmp_path,folder,seed):
    save_image = authors_save_image()
    files = sorted((AUTHORS_SHAPES/folder).glob("*.png"))[:4]
    images = {p.stem:Image.open(p) for p in files}
    names = list(images)
    positions = stimuli.place(np.random.default_rng(seed))
    for labels in ([None,"REF",None,None],config.LETTERS):
        # the authors draw both colours from the global numpy generator so seeding it fixes them
        np.random.seed(seed)
        theirs = save_image(names,images,positions,labels,str(tmp_path/"theirs.jpg"),dpi=100,use_random_colors=True)
        np.random.seed(seed)
        dark = tuple(np.random.randint(0,128) for _ in range(3))
        light = tuple(np.random.randint(180,255) for _ in range(3))
        ours = stimuli.render([images[n] for n in names],positions,labels,tmp_path/"ours.jpg",dark,light)
        assert ours==theirs
        assert (tmp_path/"ours.jpg").read_bytes()==(tmp_path/"theirs.jpg").read_bytes()


def test_place_keeps_distance_and_bounds():
    for seed in range(50):
        p = stimuli.place(np.random.default_rng(seed))
        assert p.shape==(4,2)
        assert np.all(p[:,0]>=0.3) and np.all(p[:,0]<=3.7)
        assert np.all(p[:,1]>=0.5) and np.all(p[:,1]<=2.5)
        gaps = [np.linalg.norm(p[i]-p[j]) for i in range(4) for j in range(i+1,4)]
        assert min(gaps)>=0.8


def test_shapes_have_matched_size():
    # icons are only ever drawn filled since an outline would also trace their cut out parts
    drawings = [(parts,mode) for parts in shapes.NAMED.values() for mode in ("fill","outline")]+[(parts,"fill") for parts in shapes.ICONS.values()]
    for parts,mode in drawings:
        ink = np.array(shapes.draw(parts,mode).convert("L"))<128
        ys,xs = np.nonzero(ink)
        longer = max(ys.max()-ys.min()+1,xs.max()-xs.min()+1)
        # the outline line sticks out by half its width on each side
        assert config.SHAPE_BOX-2<=longer<=config.SHAPE_BOX+config.STROKE+2
    rng = np.random.default_rng(0)
    for anchors in (5,30,50):
        ink = np.array(shapes.draw([(shapes.squiggle(rng,anchors),"add")],"open").convert("L"))<128
        ys,xs = np.nonzero(ink)
        assert config.SHAPE_BOX-2<=max(ys.max()-ys.min()+1,xs.max()-xs.min()+1)<=config.SHAPE_BOX+config.STROKE+2


def test_fill_and_outline_differ_only_in_style():
    for parts in shapes.NAMED.values():
        filled = np.array(shapes.draw(parts,"fill").convert("L"))<128
        outline = np.array(shapes.draw(parts,"outline").convert("L"))<128
        assert outline.sum()<0.5*filled.sum()
        # most of the outline sits on the edge of the filled shape
        assert (outline&filled).sum()>0.4*outline.sum()


def test_covariates_order_as_expected():
    circle = shapes.draw(shapes.NAMED["circle"],"fill")
    oval = shapes.draw(shapes.NAMED["oval"],"fill")
    star = shapes.draw(shapes.NAMED["star"],"fill")
    squiggle = shapes.draw([(shapes.squiggle(np.random.default_rng(0),30),"add")],"open")
    assert stimuli.complexity(circle)<stimuli.complexity(star)<stimuli.complexity(squiggle)
    a = stimuli.signature(circle)
    b = stimuli.signature(oval)
    c = stimuli.signature(star)
    assert stimuli.lookalike(a,[a])==pytest.approx(1.0)
    assert stimuli.lookalike(a,[b])<0.99
    assert stimuli.lookalike(a,[b])>stimuli.lookalike(a,[c])
    # only the closest wrong answer counts
    assert stimuli.lookalike(a,[b,c])==pytest.approx(stimuli.lookalike(a,[b]))


def test_build_small(tmp_path,monkeypatch):
    monkeypatch.setattr(stimuli,"STIMULI",tmp_path/"a")
    stimuli.build(8)
    monkeypatch.setattr(stimuli,"STIMULI",tmp_path/"b")
    stimuli.build(8)
    first = json.loads((tmp_path/"a"/"manifest.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path/"b"/"manifest.json").read_text(encoding="utf-8"))
    # building twice from the same seeds must give byte identical files
    assert first==second
    for name,spec in config.SETS.items():
        rows = [json.loads(l) for l in (tmp_path/"a"/name/"items.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows)==(100 if spec["kind"]=="rep" else 8)
        assert len({r["id"] for r in rows})==len(rows)
        if spec["kind"]!="rep":
            assert first["sets"][name]["letters"]=={"A":2,"B":2,"C":2,"D":2}
        for r in rows:
            assert len(set(r["shapes"]))==4
            assert r["answer"]==config.LETTERS[r["ref_index"]]
            for side in ("ref","tgt"):
                image = np.array(Image.open(tmp_path/"a"/r[f"{side}_image"]).convert("L")).astype(float)
                height,width = image.shape
                for (x0,y0),(x1,y1) in r[f"{side}_boxes"]:
                    # only the authors released sets may have shapes reaching past the picture edge
                    if spec["kind"]!="rep":
                        assert 0<=x0<x1<=width and 0<=y0<y1<=height
                    x0,y0 = max(x0,0),max(y0,0)
                    # every box must hold part of a shape which is drawn darker than the background
                    assert image[y0:y1,x0:x1].min()<image.mean()-40
