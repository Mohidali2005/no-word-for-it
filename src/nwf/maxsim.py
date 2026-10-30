import torch
import torch.nn.functional as F

from nwf import config


def maxsim(ref,targets):
    """Return for each target the mean over reference tokens of the best cosine with any of its tokens"""
    ref = F.normalize(ref,dim=-1)
    return torch.stack([(ref@F.normalize(t,dim=-1).transpose(-1,-2)).amax(dim=-1).mean(dim=-1) for t in targets],dim=-1)


def subspace(ref,targets,basis,ks):
    """Return MaxSim scores inside and outside the first k directions of one orthonormal basis"""
    return inside_outside(ref,targets,ref@basis,[t@basis for t in targets],ks)


def random_coords(x,flat,count):
    """Return coordinates of vectors in many random bases stored side by side as one wide matrix"""
    return (x@flat).reshape(len(x),count,-1).transpose(0,1)


def inside_outside(ref,targets,inner,outer,ks):
    """Return MaxSim scores inside the first k basis directions and outside them for every size k"""
    # the basis is orthonormal so the part outside it follows from the full dot products minus the inside ones
    ref_sq = (ref*ref).sum(-1)
    scores = []
    for k in ks:
        inside,outside = [],[]
        for t,full in zip(targets,outer):
            coords = full[...,:k]
            dots = inner[...,:k]@coords.transpose(-1,-2)
            ref_in = (inner[...,:k]**2).sum(-1)
            tgt_in = (coords**2).sum(-1)
            norms = ref_in.sqrt().clamp_min(1e-12)[...,:,None]*tgt_in.sqrt().clamp_min(1e-12)[...,None,:]
            inside.append((dots/norms).amax(-1).mean(-1))
            ref_out = (ref_sq-ref_in).clamp_min(0).sqrt().clamp_min(1e-12)
            tgt_out = ((t*t).sum(-1)-tgt_in).clamp_min(0).sqrt().clamp_min(1e-12)
            outside.append((((ref@t.T)-dots)/(ref_out[...,:,None]*tgt_out[...,None,:])).amax(-1).mean(-1))
        scores.append(torch.stack([torch.stack(inside,-1),torch.stack(outside,-1)],-2))
    return torch.stack(scores,-3)


def sieve(prompt,eta):
    """Fit the ridge reconstruction of vectors from the centred prompt tokens and return its centre and map"""
    p = prompt.double()
    centre = p.mean(0)
    centred = p-centre
    gram = centred@centred.T
    values = torch.linalg.eigvalsh(gram).flip(0)
    # the ridge strength is the eigenvalue where the running energy first reaches the share eta
    k = int(torch.nonzero(values.cumsum(0)>=eta*values.sum())[0])
    ridge = values[k]+1e-6*values[0]
    weights = torch.linalg.solve(gram+ridge*torch.eye(len(p),dtype=p.dtype,device=p.device),centred)
    return centre.float(),centred.float(),weights.float()


def stream_scores(ref,targets,prompt,bases,s):
    """Return the named view scores and the random view answers at one stream index"""
    named = [maxsim(ref,targets)[None]]
    for b in ("vocab","caption","image"):
        basis = bases[b] if b=="vocab" else bases[b][s]
        # doubles keep the outside part exact even when a basis holds nearly all of a vector
        named.append(subspace(ref.double(),[t.double() for t in targets],basis.double(),config.KS).reshape(-1,len(targets)).float())
    centre,centred,weights = sieve(prompt,config.SIEVE_ETA)
    ref_c = ref-centre
    tgt_c = [t-centre for t in targets]
    ref_fit = ref_c@centred.T@weights
    tgt_fit = [t@centred.T@weights for t in tgt_c]
    named.append(torch.stack([maxsim(ref_fit,tgt_fit),maxsim(ref_c-ref_fit,[a-b for a,b in zip(tgt_c,tgt_fit)])]))
    # random subspaces of the same sizes plus one the size of the prompt span which gets the same centring
    count = config.RANDOM_SUBSPACES
    assert len(prompt)-1<=bases["random"].shape[1]//count,"random subspaces are smaller than the prompt span"
    sizes = [len(ref)]+[len(t) for t in targets]
    coords = random_coords(torch.cat([ref]+targets),bases["random"],count)
    # centring is linear so the centred coordinates are the plain ones minus those of the centre
    raw_parts = torch.split(coords,sizes,1)
    centred_parts = torch.split(coords-random_coords(centre[None],bases["random"],count),sizes,1)
    plain = inside_outside(ref,targets,raw_parts[0],list(raw_parts[1:]),config.KS)
    matched = inside_outside(ref_c,tgt_c,centred_parts[0],list(centred_parts[1:]),[len(prompt)-1])
    answers = torch.cat([plain,matched],1).argmax(-1).to(torch.uint8)
    return torch.cat(named),answers
