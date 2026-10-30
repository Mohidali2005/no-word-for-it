import numpy as np
import torch

from helpers import authors_functions
from nwf import config
from nwf import maxsim


def orthonormal(d,k,seed):
    """Return a seeded random orthonormal basis"""
    return torch.from_numpy(np.linalg.qr(np.random.default_rng(seed).standard_normal((d,k)))[0])


def tokens_for(seed,d=32):
    """Return a seeded reference token set and four target sets of different sizes"""
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(7,d,generator=g,dtype=torch.float64)
    targets = [torch.randn(n,d,generator=g,dtype=torch.float64) for n in (5,9,6,8)]
    return ref,targets


def test_maxsim_matches_authors():
    theirs = authors_functions("rep_probe_qwen.py",("max_sim",),{"torch":torch})["max_sim"]
    for seed in range(20):
        ref,targets = tokens_for(seed)
        ours = maxsim.maxsim(ref.float(),[t.float() for t in targets])
        assert torch.allclose(ours,torch.tensor([theirs(ref.float(),t.float()) for t in targets]),atol=1e-6)


def test_subspace_matches_explicit_projection():
    d = 32
    ref,targets = tokens_for(0,d)
    basis = orthonormal(d,12,1)
    scores = maxsim.subspace(ref,targets,basis,[3,7,12])
    for i,k in enumerate([3,7,12]):
        u = basis[:,:k]
        inside = maxsim.maxsim(ref@u,[t@u for t in targets])
        outside = maxsim.maxsim(ref-ref@u@u.T,[t-t@u@u.T for t in targets])
        assert torch.allclose(scores[i,0],inside,atol=1e-10)
        assert torch.allclose(scores[i,1],outside,atol=1e-10)


def side_by_side(ref,targets,bases):
    """Score many bases at once the way the runs store them next to each other"""
    flat = torch.cat(list(bases),dim=1)
    coords = maxsim.random_coords(torch.cat([ref]+targets),flat,len(bases))
    parts = torch.split(coords,[len(ref)]+[len(t) for t in targets],1)
    return parts


def test_side_by_side_random_subspaces_match_one_by_one():
    d = 32
    ref,targets = tokens_for(3,d)
    ref,targets = ref.float(),[t.float() for t in targets]
    bases = [orthonormal(d,10,s).float() for s in range(5)]
    parts = side_by_side(ref,targets,bases)
    batched = maxsim.inside_outside(ref,targets,parts[0],list(parts[1:]),[4,10])
    assert batched.shape==(5,2,2,4)
    for b in range(5):
        single = maxsim.subspace(ref.double(),[t.double() for t in targets],bases[b].double(),[4,10])
        assert torch.allclose(batched[b].double(),single,atol=1e-5)


def test_projection_keeps_inside_and_drops_outside():
    d = 16
    basis = orthonormal(d,4,0)
    inside_vec = (basis@torch.randn(4,dtype=torch.float64))[None]
    outside_vec = torch.randn(1,d,dtype=torch.float64)
    outside_vec = outside_vec-outside_vec@basis@basis.T
    mixed = inside_vec+outside_vec
    scores = maxsim.subspace(mixed,[inside_vec,outside_vec],basis,[4])[0]
    # inside the subspace the mixed vector points exactly along its inside part and has nothing in common with the outside one
    # a part that is zero up to rounding turns into noise when made unit length so those entries are only near zero
    assert abs(float(scores[0,0])-1)<1e-10 and abs(float(scores[0,1]))<1e-3
    assert abs(float(scores[1,1])-1)<1e-10 and abs(float(scores[1,0]))<1e-3


def test_sieve_matches_independent_ridge_fit():
    g = torch.Generator().manual_seed(0)
    prompt = torch.randn(9,20,generator=g)*torch.linspace(3,0.2,20)
    centre,centred,weights = maxsim.sieve(prompt,config.SIEVE_ETA)
    p = prompt.double().numpy()
    tc = p-p.mean(0)
    values = np.sort(np.linalg.eigvalsh(tc@tc.T))[::-1]
    k = int(np.argmax(np.cumsum(values)>=config.SIEVE_ETA*values.sum()))
    ridge = values[k]+1e-6*values[0]
    v = torch.randn(4,20,generator=g)
    fit = ((v-centre)@centred.T@weights).double().numpy()
    for i in range(4):
        # solving the ridge problem as an ordinary least squares on stacked rows
        a = np.vstack([tc.T,np.sqrt(ridge)*np.eye(len(tc))])
        b = np.concatenate([(v[i].double().numpy()-p.mean(0)),np.zeros(len(tc))])
        coef = np.linalg.lstsq(a,b,rcond=None)[0]
        assert np.allclose(fit[i],coef@tc,atol=1e-4)


def test_stream_scores_shapes_and_random_answers(monkeypatch):
    d,s = 32,2
    ref,targets = tokens_for(5,d)
    ref,targets = ref.float(),[t.float() for t in targets]
    randoms = [orthonormal(d,8,i+20).float() for i in range(6)]
    bases = {"vocab":orthonormal(d,8,0).float(),"caption":torch.stack([orthonormal(d,8,i) for i in range(s)]).float(),"image":torch.stack([orthonormal(d,8,i+9) for i in range(s)]).float(),"random":torch.cat(randoms,dim=1)}
    monkeypatch.setattr(config,"KS",[2,4,8])
    monkeypatch.setattr(config,"RANDOM_SUBSPACES",6)
    prompt = torch.randn(6,d)
    named,answers = maxsim.stream_scores(ref,targets,prompt,bases,1)
    assert named.shape==(len(config.VIEWS),4)
    assert answers.shape==(6,4,2) and answers.dtype==torch.uint8
    assert torch.allclose(named[0],maxsim.maxsim(ref,targets))
    assert torch.isfinite(named).all()
    # each random answer equals scoring that one basis alone and the last size uses vectors centred on the prompt mean
    centre = prompt.mean(0)
    for i,basis in enumerate(randoms):
        plain = maxsim.subspace(ref.double(),[t.double() for t in targets],basis.double(),[2,4,8]).argmax(-1)
        centred = maxsim.subspace((ref-centre).double(),[(t-centre).double() for t in targets],basis.double(),[5]).argmax(-1)
        assert torch.equal(answers[i].long(),torch.cat([plain,centred]))


def test_text_share_is_zero_when_text_subspace_is_random():
    # simulated items where the right target is a noisy copy of the reference shape
    d,items,k = 48,400,8
    g = torch.Generator().manual_seed(1)
    hits = []
    bases = [orthonormal(d,k,s).float() for s in range(41)]
    for _ in range(items):
        shapes = torch.randn(4,6,d,generator=g)
        ref = shapes[0]+0.8*torch.randn(6,d,generator=g)
        parts = side_by_side(ref,list(shapes),bases)
        scores = maxsim.inside_outside(ref,list(shapes),parts[0],list(parts[1:]),[k])[:,0,0]
        hits.append((scores.argmax(-1)==0).float())
    hits = torch.stack(hits)
    # the first basis plays the text subspace and the other forty are the random baseline
    gap = hits[:,0].mean()-hits[:,1:].mean()
    assert abs(float(gap))<0.06
    assert float(hits.mean())>0.4
