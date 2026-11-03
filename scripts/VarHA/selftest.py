"""VarHA self-test. Run from the repo root:  python scripts/VarHA/selftest.py

  1. closed-form head gradient == finite differences (MSE and Huber)
  2. no leakage: corrupting every future value (truth at or after the prediction origin,
     inputs of later windows) leaves every earlier prediction bit-identical
  3. lr = 0 and no reset -> predictions equal the frozen model
  4. (needs torch) the head rebuilt from the hooked latent reproduces PhaseFormer's output
Exits non-zero on the first failure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import varha as V  # noqa: E402

rng = np.random.RandomState(0)
PL, PO, D, C, H = 24, 4, 4, 7, 96
W0, b0 = rng.randn(PO, D) * 0.3, rng.randn(PO) * 0.1
head = dict(W0=W0, b0=b0, period_len=PL, pred_len=H)
labels = np.array([0, 1, 0, 1, 2, 2, 1])


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    if not cond:
        sys.exit(1)


# 1. gradient
z, yt, mask = rng.randn(5, C, PL, D), rng.randn(5, H, C), rng.rand(5, H) > 0.3
for delta in (np.inf, 0.5):
    ln = V.VarHA(V.VarHAConfig(n_groups=3, lr=1.0, delta=delta), labels, W0, b0, PL, H)
    W, b_ = ln.W.copy(), ln.b.copy()

    def loss(Wx):
        e = V.head_forward(z, Wx, b_, labels, PL, H) - yt
        a = np.abs(e)
        hub = np.where(a <= delta, 0.5 * e ** 2, delta * (a - 0.5 * delta)) if np.isfinite(delta) else 0.5 * e ** 2
        return 2 * (hub * mask[:, :, None]).sum()

    ln.step(z, yt, mask, np.zeros(3, bool))
    num = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        Wp, Wm = W.copy(), W.copy()
        Wp[idx] += 1e-6
        Wm[idx] -= 1e-6
        num[idx] = (loss(Wp) - loss(Wm)) / 2e-6
    cnt = np.array([mask.sum() * (labels == g).sum() for g in range(3)])[:, None, None]
    ref = num * PO / cnt
    check(np.abs((W - ln.W) - ref).max() / np.abs(ref).max() < 1e-6, f"head gradient (huber delta={delta})")

# 2. causality on a synthetic stream
n = 600
st = dict(z=rng.randn(n, C, PL, D), mu=rng.randn(n, C), sd=rng.rand(n, C) + 0.5,
          y=rng.randn(n, H, C), t=np.arange(n))
cfg = V.VarHAConfig(n_groups=3, lr=0.05, batch=24, tau=1.0)
out, _ = V.run_stream(cfg, labels, st, head)
ok = True
for s in range(0, n, cfg.batch * 5):
    e = min(n, s + cfg.batch)
    bad = {k: v.copy() for k, v in st.items()}
    future = (st['t'][:, None] + np.arange(H)[None]) >= st['t'][s]     # truth at or after the origin
    bad['y'][future] = 1e6
    bad['z'][e:] = 1e6                                                   # inputs of later windows
    bad['mu'][e:] = 1e6
    o2, _ = V.run_stream(cfg, labels, bad, head)
    ok &= np.array_equal(out[:e], o2[:e])
check(ok, "no leakage: future truth / later inputs never change earlier predictions")

# 3. identity
o3, _ = V.run_stream(V.VarHAConfig(n_groups=3, lr=0.0), labels, st, head)
check(np.allclose(o3, V.base_pred(st, head)), "lr=0 reproduces the frozen model")

# 4. torch: hooked latent + numpy head == PhaseFormer output
try:
    import torch
    from argparse import Namespace
    from models.PhaseFormer import Model as PhaseFormer
except ImportError:
    print("skip  torch not installed: PhaseFormer reproduction check")
    sys.exit(0)
torch.manual_seed(0)
cfg_m = Namespace(task_name="long_term_forecast", seq_len=720, pred_len=96, enc_in=7, period_len=24,
                  phase_layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32,
                  phase_num_routers=8, phase_attn_heads=1, phase_attn_dropout=0.1, phase_use_pos_embed=1,
                  use_revin=1, revin_affine=0, revin_eps=1e-5)
m = PhaseFormer(cfg_m).eval()
cache = {}
hk = m.predictor.register_forward_hook(lambda mod, i, o: cache.__setitem__('z', i[0].detach()))
x = torch.randn(3, 720, 7).cumsum(1)
with torch.no_grad():
    y = m(x).numpy()
hk.remove()
mu = x.mean(1).numpy()
sd = torch.sqrt(x.var(1, unbiased=False) + 1e-5).numpy()
hd = dict(W0=m.predictor.decoder.weight.detach().double().numpy(),
          b0=m.predictor.decoder.bias.detach().double().numpy(), period_len=24, pred_len=96)
rec = V.base_pred(dict(z=cache['z'].double().numpy(), mu=mu, sd=sd), hd)
check(np.abs(rec - y).max() < 1e-3 * max(1.0, np.abs(y).max()), "hooked head reproduces PhaseFormer output")
