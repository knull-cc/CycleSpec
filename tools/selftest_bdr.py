"""Self-test for the BDR plug-in (models/PhaseFormerBDR.py). Run from the repo root:
    python tools/selftest_bdr.py          -> every line must say PASS
"""
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, '.')
from models import PhaseFormer as PF
from models import PhaseFormerBDR as BDR
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

RES = []


def check(name):
    def deco(fn):
        try:
            fn()
            RES.append(True)
            print('PASS', name)
        except Exception:
            RES.append(False)
            print('FAIL', name)
            print(traceback.format_exc())
        return fn
    return deco


def cfg(seq_len=720, pred_len=96, enc_in=7, period_len=24, **kw):
    c = dict(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, period_len=period_len, latent_dim=4,
             phase_encoder_hidden=16, predictor_hidden=32, phase_layers=3, phase_attn_heads=1,
             phase_attn_dropout=0.1, phase_attn_use_relpos=True, phase_num_routers=8,
             phase_use_pos_embed=True, use_revin=True, revin_affine=False, revin_eps=1e-5,
             fix_seed=2021, bdr_stages=3, bdr_hidden=64, bdr_ctx=96)
    c.update(kw)
    return SimpleNamespace(**c)


@check('same global RNG stream and identical backbone init as plain PhaseFormer (same seed)')
def _():
    torch.manual_seed(2021)
    pf = PF.Model(cfg())
    a = torch.rand(5)
    torch.manual_seed(2021)
    bd = BDR.Model(cfg())
    b = torch.rand(5)
    assert torch.equal(a, b), 'refiner init consumed global RNG'
    sp, sb = pf.state_dict(), bd.state_dict()
    for k in sp:
        assert torch.equal(sp[k], sb[k]), k


@check('at init the plug-in is the identity: output == PhaseFormer output')
def _():
    torch.manual_seed(2021)
    pf = PF.Model(cfg()).eval()
    torch.manual_seed(2021)
    bd = BDR.Model(cfg()).eval()
    x = torch.randn(4, 720, 7)
    with torch.no_grad():
        assert torch.allclose(pf(x), bd(x), atol=1e-6)


@check('shapes / stage lengths: ETTh1 H=96..720 and Exchange (seq 96, period 7)')
def _():
    for c, x in [(cfg(pred_len=H), torch.randn(3, 720, 7)) for H in (96, 192, 336, 720)] + \
                [(cfg(seq_len=96, pred_len=H, enc_in=8, period_len=7, phase_layers=1, latent_dim=8,
                      phase_encoder_hidden=32, predictor_hidden=64, phase_num_routers=4), torch.randn(3, 96, 8))
                 for H in (96, 192, 336, 720)]:
        m = BDR.Model(c).train()
        out = m(x)
        assert out.shape == (3, c.pred_len, c.enc_in), out.shape
        lens = [o.shape[1] for o in m.stage_outputs]
        assert lens[-1] == c.pred_len and lens == sorted(lens), lens
        m.eval()
        with torch.no_grad():
            m(x)
        assert m.stage_outputs is None


@check('training: loss decreases, refiner and backbone both get gradients (w_mode last and equal)')
def _():
    torch.manual_seed(0)
    t = torch.arange(816).float()
    base = torch.sin(t * 2 * 3.14159 / 24)[None, :, None].repeat(8, 1, 7) + 0.1 * torch.randn(8, 816, 7)
    xb, yb = base[:, :720], base[:, 720:]
    for mode in ('last', 'equal'):
        m = BDR.Model(cfg()).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        me = SimpleNamespace(model=m, args=SimpleNamespace(bdo_w_mode=mode))
        losses = []
        for i in range(30):
            opt.zero_grad()
            loss = Exp_Long_Term_Forecast._train_loss(me, m(xb), yb, nn.MSELoss(), 0)
            loss.backward()
            if i == 1:
                g_ref = sum(p.grad.abs().sum() for p in m.refiner.parameters() if p.grad is not None)
                g_bb = m.embedding.projection.weight.grad.abs().sum()
                assert g_ref > 0 and g_bb > 0, (mode, float(g_ref), float(g_bb))
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (mode, losses[0], losses[-1])


n_pf = sum(p.numel() for p in PF.Model(cfg()).parameters())
for H in (96, 720):
    m = BDR.Model(cfg(pred_len=H))
    print('ETTh1 H=%d: PhaseFormer %d params, + BDR refiner %d params' % (
        H, sum(p.numel() for p in PF.Model(cfg(pred_len=H)).parameters()), sum(p.numel() for p in m.refiner.parameters())))
print('%d / %d passed' % (sum(RES), len(RES)))
sys.exit(0 if all(RES) else 1)
