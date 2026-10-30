import math

import torch


def qwen_tokens(input_ids,grid,size,box,image_index,start_id,end_id):
    """Return the positions of the Qwen tokens whose 32 pixel block overlaps a shape box"""
    ids = input_ids.tolist()
    starts = [i for i,t in enumerate(ids) if t==start_id]
    first = starts[image_index]+1
    count = ids.index(end_id,first)-first
    # the processor resizes each side to a multiple of 32 so the box is scaled the way the authors did
    height,width = int(grid[1])*16,int(grid[2])*16
    sx,sy = width/size[0],height/size[1]
    x0,y0,x1,y1 = int(box[0][0]*sx),int(box[0][1]*sy),int(box[1][0]*sx),int(box[1][1]*sy)
    rows,cols = height//32,width//32
    assert rows*cols==count,"token grid does not match the image"
    positions = []
    for gy in range(max(0,y0//32),min(rows,(y1+31)//32)):
        for gx in range(max(0,x0//32),min(cols,(x1+31)//32)):
            if gx*32<x1 and (gx+1)*32>x0 and gy*32<y1 and (gy+1)*32>y0:
                positions.append(first+gy*cols+gx)
    return positions


def gemma_crops(width,height,min_size,max_crops,min_ratio):
    """Return the pan and scan crop boxes the Gemma processor cuts from an image"""
    if width>=height:
        if width/height<min_ratio:
            return []
        across = max(2,min(max_crops,min(int(math.floor(width/min_size)),int(math.floor(width/height+0.5)))))
        down = 1
    else:
        if height/width<min_ratio:
            return []
        down = max(2,min(max_crops,min(int(math.floor(height/min_size)),int(math.floor(height/width+0.5)))))
        across = 1
    cw,ch = math.ceil(width/across),math.ceil(height/down)
    if min(cw,ch)<min_size:
        return []
    return [(cw*i,ch*j,min(cw*i+cw,width),min(ch*j+ch,height)) for j in range(down) for i in range(across)]


def gemma_tokens(input_ids,crops,box,first_slot,start_id,side=896,per_side=16):
    """Return the positions of the Gemma crop tokens that cover a shape box"""
    starts = [i for i,t in enumerate(input_ids.tolist()) if t==start_id]
    patch = side/per_side
    positions = []
    for c,(cx0,cy0,cx1,cy1) in enumerate(crops):
        ix0,iy0 = max(box[0][0],cx0),max(box[0][1],cy0)
        ix1,iy1 = min(box[1][0],cx1),min(box[1][1],cy1)
        if ix0>=ix1 or iy0>=iy1:
            continue
        # each crop is stretched to a square so x and y get their own scale
        sx,sy = side/(cx1-cx0),side/(cy1-cy0)
        tx0,ty0 = int((ix0-cx0)*sx//patch),int((iy0-cy0)*sy//patch)
        tx1 = max(int((ix1-cx0)*sx//patch),tx0+1)
        ty1 = max(int((iy1-cy0)*sy//patch),ty0+1)
        tx0,ty0 = max(0,min(tx0,per_side-1)),max(0,min(ty0,per_side-1))
        tx1,ty1 = max(0,min(tx1,per_side)),max(0,min(ty1,per_side))
        offset = starts[first_slot+c]+1
        positions += [offset+y*per_side+x for y in range(ty0,ty1) for x in range(tx0,tx1)]
    return positions


def after_images(input_ids,end_id):
    """Return the positions of every token after the last image"""
    ids = input_ids.tolist()
    last = max(i for i,t in enumerate(ids) if t==end_id)
    return list(range(last+1,len(ids)))


def image_positions(input_ids,image_id):
    """Return the positions of every image token"""
    return torch.nonzero(input_ids==image_id).flatten().tolist()
