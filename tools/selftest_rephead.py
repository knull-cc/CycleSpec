"""Self-test for the re-parameterised head. Run from the repo root before training:
    python tools/selftest_rephead.py
Every line must say PASS. Written without torch available here: send me any traceback.
"""
import copy
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, '.')
from layers.RepHead import RepLinear, cascade_sizes
from models import PhaseFormer as PF
from models import PhaseFormerRep as REP

RESULTS = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append(True)
            print('PASS', name)
        except Exception:
            RESULTS.append(False)
            print('FAIL', name)
            print(traceback.format_exc())
        return fn
    return deco


def cfg(pred_len=96, **kw):
    c = dict(seq_len=720, pred_len=pred_len, enc_in=7, period_len=24, latent_dim=4,
             phase_encoder_hidden=16, predictor_hidden=32, phase_layers=3, phase_attn_heads=1,
             phase_attn_dropout=0.0, phase_attn_use_relpos=True, phase_num_routers=8,
             phase_use_pos_embed=True, use_revin=True, revin_affine=False, revin_eps=1e-5,
             rep_mode='chain', rep_depth=2, rep_width=0, rep_stages=3)
    c.update(kw)
    return SimpleNamespace(**c)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


torch.manual_seed(0)
X = torch.randn(4, 720, 7)
MODES = [dict(mode='linear'), dict(mode='chain', depth=2), dict(mode='chain', depth=3, width=32),
         dict(mode='cascade', stages=3)]


@check('cascade sizes: P_out=4 -> [2,3,4], 8 -> [3,6,8], 30 -> [10,20,30]')
def _():
    assert cascade_sizes(4, 3) == [2, 3, 4] and cascade_sizes(8, 3) == [3, 6, 8] and cascade_sizes(30, 3) == [10, 20, 30]


@check('RepLinear.merged() reproduces the over-parameterised map (all modes, 4-d input)')
def _():
    z = torch.randn(3, 7, 24, 4, dtype=torch.float64)
    for kw in MODES:
        r = RepLinear(4, 4, **kw).double()
        for p in r.parameters():
            nn.init.normal_(p, std=0.5)
        m = r.merged()
        assert isinstance(m, nn.Linear) and m.weight.shape == (4, 4)
        assert torch.allclose(r(z), m(z), atol=1e-10), (kw, (r(z) - m(z)).abs().max())


@check('PhaseFormerRep rep_mode=linear with copied weights == PhaseFormer')
def _():
    pf = PF.Model(cfg()).eval()
    rp = REP.Model(cfg(rep_mode='linear')).eval()
    sd = copy.deepcopy(pf.state_dict())
    sd['predictor.decoder.layers.0.weight'] = sd.pop('predictor.decoder.weight')
    sd['predictor.decoder.layers.0.bias'] = sd.pop('predictor.decoder.bias')
    rp.load_state_dict(sd, strict=True)
    with torch.no_grad():
        assert torch.allclose(pf(X), rp(X), atol=1e-6)


@check('shapes for H = 96, 192, 336, 720, every mode')
def _():
    for H in (96, 192, 336, 720):
        for kw in MODES:
            c = cfg(H, rep_mode=kw['mode'], rep_depth=kw.get('depth', 2), rep_width=kw.get('width', 0))
            m = REP.Model(c).eval()
            with torch.no_grad():
                assert m(X).shape == (4, H, 7)


@check('merge_head(): same predictions, PhaseFormer parameter count, loads into plain PhaseFormer')
def _():
    for kw in MODES[1:]:
        c = cfg(rep_mode=kw['mode'], rep_depth=kw.get('depth', 2), rep_width=kw.get('width', 0))
        m = REP.Model(c).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(3):  # move away from init
            opt.zero_grad()
            m(X).pow(2).mean().backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            a = m(X)
            m.merge_head()
            b = m(X)
        assert torch.allclose(a, b, atol=1e-5), (kw, (a - b).abs().max())
        pf = PF.Model(cfg())
        assert nparams(m) == nparams(pf), (nparams(m), nparams(pf))
        pf.load_state_dict(m.state_dict(), strict=True)  # merged model IS a PhaseFormer
        pf.eval()
        with torch.no_grad():
            assert torch.allclose(pf(X), b, atol=1e-6)


@check('gradients reach every head layer and the embedding')
def _():
    for kw in MODES[1:]:
        c = cfg(rep_mode=kw['mode'], rep_depth=kw.get('depth', 2), rep_width=kw.get('width', 0))
        m = REP.Model(c).train()
        m(X).pow(2).mean().backward()
        for i, l in enumerate(m.predictor.decoder.layers):
            assert l.weight.grad is not None and l.weight.grad.abs().sum() > 0, (kw, i)
        assert m.embedding.projection.weight.grad.abs().sum() > 0


@check('30 Adam steps reduce the loss (every mode)')
def _():
    torch.manual_seed(1)
    t = torch.arange(816).float()
    base = torch.sin(t * 2 * 3.14159 / 24)[None, :, None].repeat(8, 1, 7) + 0.05 * torch.randn(8, 816, 7)
    xb, yb = base[:, :720], base[:, 720:]
    for kw in MODES:
        c = cfg(rep_mode=kw['mode'], rep_depth=kw.get('depth', 2), rep_width=kw.get('width', 0))
        m = REP.Model(c).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = nn.MSELoss()(m(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (kw, losses[0], losses[-1])


print()
for kw in MODES:
    c = cfg(rep_mode=kw['mode'], rep_depth=kw.get('depth', 2), rep_width=kw.get('width', 0))
    print('train-time params %-34s %d' % (kw, nparams(REP.Model(c))))
print('PhaseFormer (= every merged model)                  %d' % nparams(PF.Model(cfg())))
print('%d / %d passed' % (sum(RESULTS), len(RESULTS)))
sys.exit(0 if all(RESULTS) else 1)
