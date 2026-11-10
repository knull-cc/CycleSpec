"""Self-test for the EMA / BDO plugin. Run from the repo root BEFORE any training:
    python tools/selftest_plugin.py
Every line must say PASS. (Written without being able to run torch, so a FAIL here may be a bug
in this test as much as in the code: send me the traceback.)
"""
import copy
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, '.')
from models import PhaseFormer as PF
from models import PhaseFormerBDO as BDO
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.ema import ModelEMA

RESULTS = []


def check(name):
    def deco(fn):
        try:
            fn()
            RESULTS.append((name, True, ''))
            print('PASS', name)
        except Exception:
            RESULTS.append((name, False, traceback.format_exc()))
            print('FAIL', name)
            print(traceback.format_exc())
        return fn
    return deco


def cfg(pred_len=96, **kw):
    # ETTh1 official config for H<=336
    c = dict(seq_len=720, pred_len=pred_len, enc_in=7, period_len=24, latent_dim=4,
             phase_encoder_hidden=16, predictor_hidden=32, phase_layers=3, phase_attn_heads=1,
             phase_attn_dropout=0.0, phase_attn_use_relpos=True, phase_num_routers=8,
             phase_use_pos_embed=True, use_revin=True, revin_affine=False, revin_eps=1e-5,
             bdo_stages=3)
    c.update(kw)
    return SimpleNamespace(**c)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


torch.manual_seed(0)
X = torch.randn(4, 720, 7)


@check('BDO stages=1 reproduces PhaseFormer exactly (same weights)')
def _():
    pf = PF.Model(cfg()).eval()
    bd = BDO.Model(cfg(bdo_stages=1)).eval()
    sd = copy.deepcopy(pf.state_dict())
    sd['heads.0.weight'] = sd.pop('predictor.decoder.weight')
    sd['heads.0.bias'] = sd.pop('predictor.decoder.bias')
    missing, unexpected = bd.load_state_dict(sd, strict=True), None
    assert nparams(pf) == nparams(bd), (nparams(pf), nparams(bd))
    with torch.no_grad():
        a, b = pf(X), bd(X)
    assert a.shape == b.shape == (4, 96, 7)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


@check('BDO shapes / stage prefixes for H = 96, 192, 336, 720')
def _():
    for H, want in [(96, [48, 72, 96]), (192, [72, 144, 192]), (336, [120, 240, 336]), (720, [240, 480, 720])]:
        m = BDO.Model(cfg(H, bdo_stages=3)).train()
        out = m(X)
        assert out.shape == (4, H, 7), out.shape
        lens = [o.shape[1] for o in m.stage_outputs]
        assert lens == want, (H, lens, want)
        assert torch.equal(out, m.stage_outputs[-1])
        m.eval()
        m(X)
        assert m.stage_outputs is None


@check('stage_outputs only kept in training mode; eval output deterministic')
def _():
    m = BDO.Model(cfg()).eval()
    with torch.no_grad():
        assert torch.equal(m(X), m(X))


@check('multi-stage loss: all stages weighted, gradients reach heads and backbone')
def _():
    m = BDO.Model(cfg()).train()
    y = torch.randn(4, 96, 7)
    out = m(X)
    me = SimpleNamespace(model=m, args=SimpleNamespace(bdo_w_mode='inv'))
    loss = Exp_Long_Term_Forecast._train_loss(me, out, y, nn.MSELoss(), 0)
    loss.backward()
    for i, h in enumerate(m.heads):
        assert h.weight.grad is not None and h.weight.grad.abs().sum() > 0, 'head %d got no gradient' % i
    assert m.embedding.projection.weight.grad.abs().sum() > 0


@check('w_mode=last equals plain criterion on the final output')
def _():
    m = BDO.Model(cfg()).train()
    y = torch.randn(4, 96, 7)
    out = m(X)
    me = SimpleNamespace(model=m, args=SimpleNamespace(bdo_w_mode='last'))
    l1 = Exp_Long_Term_Forecast._train_loss(me, out, y, nn.MSELoss(), 0)
    l2 = nn.MSELoss()(out, y)
    assert torch.allclose(l1, l2, atol=1e-7)


@check('plain PhaseFormer: _train_loss falls back to criterion')
def _():
    m = PF.Model(cfg()).train()
    y = torch.randn(4, 96, 7)
    out = m(X)
    me = SimpleNamespace(model=m, args=SimpleNamespace(bdo_w_mode='inv'))
    assert torch.allclose(Exp_Long_Term_Forecast._train_loss(me, out, y, nn.MSELoss(), 0), nn.MSELoss()(out, y))


@check('EMA: math, independence from the live model, buffers copied')
def _():
    lin = nn.Linear(2, 2, bias=False)
    nn.init.constant_(lin.weight, 1.0)
    ema = ModelEMA(lin, decay=0.5, warmup=False)
    with torch.no_grad():
        lin.weight.fill_(3.0)
    ema.update(lin)  # 0.5*1 + 0.5*3 = 2
    assert torch.allclose(ema.module.weight, torch.full((2, 2), 2.0))
    ema.update(lin)  # 0.5*2 + 0.5*3 = 2.5
    assert torch.allclose(ema.module.weight, torch.full((2, 2), 2.5))
    assert ema.module.weight.requires_grad is False
    assert lin.weight.requires_grad is True
    ema2 = ModelEMA(lin, decay=0.996, warmup=True)
    ema2.update(lin)
    assert abs(ema2.current_decay() - (1 + 1) / (10 + 1)) < 1e-9


@check('EMA on a full BDO model runs and keeps state_dict keys identical')
def _():
    m = BDO.Model(cfg()).train()
    ema = ModelEMA(m, decay=0.9)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    y = torch.randn(4, 96, 7)
    for _ in range(3):
        opt.zero_grad()
        nn.MSELoss()(m(X), y).backward()
        opt.step()
        ema.update(m)
    assert list(ema.module.state_dict().keys()) == list(m.state_dict().keys())
    m.load_state_dict(ema.module.state_dict())  # what exp.train() does at the end


@check('30 optimiser steps reduce the multi-stage loss (BDO and PhaseFormer)')
def _():
    torch.manual_seed(1)
    t = torch.arange(720 + 96).float()
    base = torch.sin(t * 2 * 3.14159 / 24)[None, :, None].repeat(8, 1, 7) + 0.05 * torch.randn(8, 816, 7)
    xb, yb = base[:, :720], base[:, 720:]
    for Mod in (PF, BDO):
        m = Mod.Model(cfg()).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        me = SimpleNamespace(model=m, args=SimpleNamespace(bdo_w_mode='inv'))
        first = last = None
        for i in range(30):
            opt.zero_grad()
            loss = Exp_Long_Term_Forecast._train_loss(me, m(xb), yb, nn.MSELoss(), 0)
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
        assert last < first, (Mod.__name__, first, last)


print()
print('parameters (ETTh1 H=96 official config): PhaseFormer %d | BDO n=1 %d | BDO n=3 %d' % (
    nparams(PF.Model(cfg())), nparams(BDO.Model(cfg(bdo_stages=1))), nparams(BDO.Model(cfg(bdo_stages=3)))))
bad = [r for r in RESULTS if not r[1]]
print('%d / %d passed' % (len(RESULTS) - len(bad), len(RESULTS)))
sys.exit(1 if bad else 0)
