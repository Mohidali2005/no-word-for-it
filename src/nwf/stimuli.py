import argparse
import hashlib
import io
import json
import shutil

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from scipy.ndimage import gaussian_filter

from nwf import config
from nwf import shapes
from nwf.paths import AUTHORS_SHAPES
from nwf.paths import STIMULI


def recolor(image,dark,light):
    """Paint the dark parts of an image in one colour and the light parts in another while keeping the red labels"""
    rgb = np.array(image.convert("RGB")).astype(float)
    red = (rgb[:,:,0]>200)&(rgb[:,:,1]<100)&(rgb[:,:,2]<100)
    gray = np.array(image.convert("RGB").convert("L")).astype(float)/255.0
    # keeping the authors exact order of operations so the pixels come out identical
    out = np.stack([(1-gray)*dark[c]+gray*light[c] for c in range(3)],axis=2).astype(np.uint8)
    out[red] = rgb[red].astype(np.uint8)
    return Image.fromarray(out)


def place(rng):
    """Pick four shape centres that keep the authors minimum distance of 0.8"""
    positions = []
    for _ in range(4):
        for _ in range(1000):
            # the authors let centres reach 0.25 from the side so shapes could stick out of the picture
            # keeping them 0.3 from the side means every shape is drawn whole
            pos = rng.random(2)*[3.4,2.0]+[0.3,0.5]
            if all(np.linalg.norm(pos-p)>=0.8 for p in positions):
                positions.append(pos)
                break
    # the authors silently dropped a shape when no spot was found so this stops instead
    assert len(positions)==4,"could not place four shapes"
    return np.array(positions)


def render(images,positions,labels,path,dark,light):
    """Draw four shapes with their labels the way the authors did and return each shape box in pixels"""
    fig = Figure(figsize=(8,6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_xlim(0,4)
    ax.set_ylim(0,3)
    ax.axis("off")
    extents = []
    for image,pos,label in zip(images,positions,labels):
        extent = [pos[0]-0.3,pos[0]+0.3,pos[1]-0.3,pos[1]+0.3]
        extents.append(extent)
        ax.imshow(image,extent=extent,aspect="auto",zorder=1)
        if label:
            ax.text(pos[0],pos[1]-0.40,label,fontsize=20,ha="center",weight="bold",color="red",zorder=2)
    fig.tight_layout(pad=0)
    # saving as jpeg twice like the authors who wrote the figure and then the recoloured copy
    buffer = io.BytesIO()
    fig.savefig(buffer,format="jpg",dpi=100,bbox_inches="tight",pad_inches=0)
    saved = Image.open(buffer)
    recolor(saved,dark,light).save(path)
    width,height = saved.size

    canvas.draw()
    tight = fig.get_tightbbox(canvas.get_renderer())
    box = ax.get_position()
    figw,figh = fig.get_size_inches()
    left = (box.x0*figw-tight.x0)*100
    bottom = (box.y0*figh-tight.y0)*100
    axw = box.width*figw*100
    axh = box.height*figh*100
    top = height-(bottom+axh)
    boxes = []
    for x0,x1,y0,y1 in extents:
        boxes.append([[int(round(left+x0*(axw/4.0))),int(round(top+(3.0-y1)*(axh/3.0)))],[int(round(left+x1*(axw/4.0))),int(round(top+(3.0-y0)*(axh/3.0)))]])
    return boxes


def complexity(image):
    """Return perimeter squared over area of the ink which is lowest for a filled circle"""
    ink = np.array(image.convert("L").resize((config.CANVAS,config.CANVAS),Image.LANCZOS))<128
    padded = np.pad(ink,1)
    perimeter = np.sum(padded[1:,:]!=padded[:-1,:])+np.sum(padded[:,1:]!=padded[:,:-1])
    return float(perimeter**2/(4*np.pi*ink.sum()))


def signature(image):
    """Return a blurred and size normalised ink mask used to judge how alike two shapes look"""
    ink = (np.array(image.convert("L"))<128).astype(np.float32)
    ys,xs = np.nonzero(ink)
    crop = ink[ys.min():ys.max()+1,xs.min():xs.max()+1]
    # padding the crop to a square with a margin so a long shape keeps its proportions and a filled square still has an edge
    side = max(crop.shape)*3//2
    square = np.zeros((side,side),np.float32)
    y0 = (side-crop.shape[0])//2
    x0 = (side-crop.shape[1])//2
    square[y0:y0+crop.shape[0],x0:x0+crop.shape[1]] = crop
    small = np.array(Image.fromarray(square).resize((64,64),Image.BILINEAR))
    blurred = gaussian_filter(small,3)
    return (blurred-blurred.mean()).ravel()


def lookalike(ref,others):
    """Return the highest correlation between the reference signature and any wrong answer signature"""
    return max(float(ref@o/(np.linalg.norm(ref)*np.linalg.norm(o))) for o in others)


def sha(path):
    """Return the sha256 of a file"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(items):
    """Make every stimulus set with its images items file and a manifest of hashes"""
    STIMULI.mkdir(parents=True,exist_ok=True)
    manifest = {"authors_commit":config.AUTHORS_COMMIT,"items_per_set":items,"sets":{}}
    for name,spec in config.SETS.items():
        folder = STIMULI/name
        if folder.exists():
            shutil.rmtree(folder)
        (folder/"shapes").mkdir(parents=True)

        rng = np.random.default_rng(spec.get("seed",0))
        kind = spec["kind"]
        if kind=="named":
            pool = {n:shapes.draw(parts,"fill" if spec["filled"] else "outline") for n,parts in shapes.NAMED.items()}
        elif kind=="icon":
            pool = {n:shapes.draw(parts,"fill") for n,parts in shapes.ICONS.items()}
        elif kind=="blob":
            pool = {f"blob_{i:02d}":shapes.draw([(shapes.blob(rng),"add")],"fill") for i in range(config.POOL)}
        elif kind=="squiggle":
            pool = {f"squig{spec['anchors']}_{i:02d}":shapes.draw([(shapes.squiggle(rng,spec["anchors"]),"add")],"open") for i in range(config.POOL)}
        else:
            # the authors images are used exactly as they load them so nothing is converted here
            pool = {p.stem:Image.open(p) for p in sorted((AUTHORS_SHAPES/spec["shapes" if kind=="rep" else "folder"]).glob("*.png"))}
        for n,image in pool.items():
            image.save(folder/"shapes"/f"{n}.png")
        pixels = {hashlib.sha256(np.array(image).tobytes()).hexdigest() for image in pool.values()}
        assert len(pixels)==len(pool),f"{name} has two identical shapes"
        signatures = {n:signature(image) for n,image in pool.items()}
        scores = {n:complexity(image) for n,image in pool.items()}

        rows = []
        if kind=="rep":
            answers = json.loads((AUTHORS_SHAPES/spec["folder"]/"answers.json").read_text(encoding="utf-8"))
            for a in answers:
                item = f"{name}_{a['id']:04d}"
                # the answers file stores paths relative to the authors folder
                shutil.copyfile(AUTHORS_SHAPES/a["ref_image_path"],folder/f"{item}_ref.jpg")
                shutil.copyfile(AUTHORS_SHAPES/a["tgt_image_path"],folder/f"{item}_tgt.jpg")
                rows.append({"id":item,"shapes":a["shape_types"],"ref_index":ord(a["answer"])-65,"ref_boxes":a["ref_pixel_positions"],"tgt_boxes":a["tgt_pixel_positions"]})
        else:
            ids = list(pool)
            # splitting big pools in two halves so a probe can be tested on shapes it never saw
            halves = [ids[0::2],ids[1::2]] if len(ids)>=config.POOL else [ids,ids]
            # cycling through the four letters before shuffling so every letter is the answer equally often
            order = np.random.default_rng([spec["seed"],10**6]).permutation(np.tile(np.arange(4),items//4+1)[:items])
            for i in range(items):
                r = np.random.default_rng([spec["seed"],i])
                item = f"{name}_{i:04d}"
                chosen = [str(c) for c in r.choice(halves[i%2],4,replace=False)]
                ref = int(order[i])
                labels = [None]*4
                labels[ref] = "REF"
                images = [pool[c] for c in chosen]
                ref_boxes = render(images,place(r),labels,folder/f"{item}_ref.jpg",r.integers(0,128,3),r.integers(180,255,3))
                tgt_boxes = render(images,place(r),config.LETTERS,folder/f"{item}_tgt.jpg",r.integers(0,128,3),r.integers(180,255,3))
                rows.append({"id":item,"shapes":chosen,"ref_index":ref,"ref_boxes":ref_boxes,"tgt_boxes":tgt_boxes})

        for k,row in enumerate(rows):
            row["set"] = name
            row["answer"] = config.LETTERS[row["ref_index"]]
            row["ref_image"] = f"{name}/{row['id']}_ref.jpg"
            row["tgt_image"] = f"{name}/{row['id']}_tgt.jpg"
            row["named"] = spec["named"]
            row["filled"] = spec["filled"]
            row["complexity"] = [scores[s] for s in row["shapes"]]
            # the closest wrong answer is what makes a choice hard so only that one counts
            ref = row["shapes"][row["ref_index"]]
            row["lookalike"] = lookalike(signatures[ref],[signatures[s] for s in row["shapes"] if s!=ref])
            row["calibration"] = k<config.CALIBRATION

        ids = [row["id"] for row in rows]
        assert len(set(ids))==len(ids),f"{name} has duplicate ids"
        # writing plain newlines so the files are byte identical on windows and linux
        with open(folder/"items.jsonl","w",encoding="utf-8",newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row)+"\n")
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        listing = "".join(f"{p.relative_to(STIMULI).as_posix()} {sha(p)}\n" for p in files)
        letters = [row["answer"] for row in rows]
        manifest["sets"][name] = {
            "items":len(rows),
            "shapes":len(pool),
            "files":len(files),
            "letters":{l:letters.count(l) for l in config.LETTERS},
            "items_sha256":sha(folder/"items.jsonl"),
            "all_files_sha256":hashlib.sha256(listing.encode()).hexdigest(),
        }
        print(f"{name}: {len(rows)} items from {len(pool)} shapes, letters {manifest['sets'][name]['letters']}")

    (STIMULI/"manifest.json").write_text(json.dumps(manifest,indent=1),encoding="utf-8",newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build the stimulus sets")
    parser.add_argument("--items",type=int,default=config.ITEMS,help="items per generated set")
    args = parser.parse_args()
    build(args.items)
