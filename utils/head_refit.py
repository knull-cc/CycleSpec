"""N+ : post-hoc closed-form refit of PhaseFormer's prediction head (no retraining, no extra inference cost).

After training, freeze everything up to the head, collect the head inputs on the TRAIN split and solve
the head by weighted ridge least squares in closed form. The weights make the objective exactly the
test metric: RevIN de-normalisation multiplies the error by sigma, so the MSE on the (standardised)
data is sum sigma^2 * err_n^2.

Variants (all chosen on VALIDATION only; TEST is reported once):
  features  Z    : the original head input (latent, D dims)               -> same function class as PF
            Z+X  : latent + the raw (RevIN-normalised) input phase series -> adds a linear input skip
  groups    shared | phase (one head per phase) | chan (per channel) | phase_chan (per phase and channel)
  ridge     lambda_rel in LAMBDAS (relative to the mean diagonal of the normal matrix; bias not penalised)

Writes results/<setting>/refit.json and prints base vs refit metrics. Works for PhaseFormer and
PhaseFormerRep (any model with PhaseFormer's embedding / routing_layers / predictor).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GROUPS = ['shared', 'phase', 'chan', 'phase_chan']
FEATS = ['Z', 'Z+X']


@torch.no_grad()
def _encode(model, x_enc):
    """PhaseFormer steps 1-8: returns head input Z (B,C,L,D), input phase series (B,C,L,P_in), RevIN stats."""
    if model.use_revin:
        x_in, stats = model.revin.normalize(x_enc)
    else:
        x_in = x_enc.float()
        stats = (torch.zeros_like(x_in[:, :1]), torch.ones_like(x_in[:, :1]))
    x = x_in.permute(0, 2, 1)
    B, C, _ = x.shape
    if model.pad_seq_len > 0:
        x = F.pad(x, (0, model.pad_seq_len), mode='circular')
    ps = model._to_phase_series(x.view(B, C, model.num_periods_input, model.period_len))
    Z, cur = model.embedding(ps), ps
    for i, unit in enumerate(model.routing_layers):
        Z, y = unit(cur, Z)
        if i < len(model.routing_layers) - 1:
            cur = y
    return Z, ps, stats


@torch.no_grad()
def _batch(model, batch_x, batch_y, pred_len, period_len, device):
    """Features X (B,C,L,F) with bias column last, normalised targets T (B,C,L,P), weights w=sigma^2 (B,C)."""
    batch_x = batch_x.float().to(device)
    y = batch_y[:, -pred_len:, :].float().to(device)  # (B,H,C), standardised data
    Z, ps, (mu, sigma) = _encode(model, batch_x)
    B, C, L, _ = Z.shape
    tn = ((y - mu) / sigma).permute(0, 2, 1)  # (B,C,H)
    T = tn.reshape(B, C, pred_len // period_len, period_len).permute(0, 1, 3, 2)  # (B,C,L,P_out)
    ones = torch.ones(B, C, L, 1, device=device)
    X = torch.cat([Z, ps, ones], dim=-1).double()
    w = (sigma[:, 0, :] ** 2).double()  # (B,C)
    base = model.predictor(Z)  # (B,C,L,P_out), original head
    return X, T.double(), w, base.double()


def _feat_idx(name, D, P_in):
    idx = list(range(D)) + (list(range(D, D + P_in)) if name == 'Z+X' else [])
    return idx + [D + P_in]  # bias last


def _group_sum(A, g):
    """A: (C,L,...) normal-equation blocks -> grouped (G...) and an expander back to (C,L)."""
    C, L = A.shape[:2]
    if g == 'phase_chan':
        return A
    if g == 'phase':
        return A.sum(0, keepdim=True).expand(C, *A.shape[1:])
    if g == 'chan':
        return A.sum(1, keepdim=True).expand(C, L, *A.shape[2:])
    return A.sum((0, 1), keepdim=True).expand(C, L, *A.shape[2:])


def head_refit(exp, setting):
    args = exp.args
    model = exp.model.module if isinstance(exp.model, nn.DataParallel) else exp.model
    model.eval()
    device = exp.device
    H, Lp = args.pred_len, model.period_len
    if H % Lp != 0:
        print('[refit] skipped: pred_len %d is not a multiple of period_len %d' % (H, Lp))
        return None
    D, P_in = model.latent_dim, model.num_periods_input

    # 1) normal equations on TRAIN, per (channel, phase)
    _, train_loader = exp._get_data(flag='train')
    AtA = AtB = None
    for batch_x, batch_y, _, _ in train_loader:
        X, T, w, _ = _batch(model, batch_x, batch_y, H, Lp, device)
        Xw = X * w[:, :, None, None]
        a = torch.einsum('bclf,bclg->clfg', Xw, X)
        b = torch.einsum('bclf,bclp->clfp', Xw, T)
        AtA = a if AtA is None else AtA + a
        AtB = b if AtB is None else AtB + b

    # 2) solve every (feature set, grouping, lambda)
    combos, Ws = [], []
    for fs in FEATS:
        idx = torch.tensor(_feat_idx(fs, D, P_in), device=device)
        A = AtA[:, :, idx][:, :, :, idx]
        Bm = AtB[:, :, idx]
        for g in GROUPS:
            Ag, Bg = _group_sum(A, g), _group_sum(Bm, g)
            nF = len(idx)
            scale = torch.diagonal(Ag, dim1=-2, dim2=-1)[..., :-1].mean(-1).clamp_min(1e-12)  # (C,L)
            R = torch.eye(nF, device=device, dtype=Ag.dtype)
            R[-1, -1] = 0.0  # do not penalise the bias
            for lam in LAMBDAS:
                M = Ag + ((lam + 1e-9) * scale)[..., None, None] * R  # tiny jitter keeps lam=0 solvable
                Wc = torch.linalg.solve(M, Bg)  # (C,L,nF,P)
                Wfull = torch.zeros(*Wc.shape[:2], D + P_in + 1, Wc.shape[-1], device=device, dtype=Wc.dtype)
                Wfull[:, :, idx] = Wc
                combos.append((fs, g, lam))
                Ws.append(Wfull)
    Wst = torch.stack(Ws)  # (K,C,L,F,P)

    # 3) evaluate base head and every combo on VAL and TEST (metrics on the standardised data, like exp.test)
    def evaluate(flag):
        _, loader = exp._get_data(flag=flag)
        K = Wst.shape[0]
        se, ae = torch.zeros(K, device=device, dtype=torch.float64), torch.zeros(K, device=device, dtype=torch.float64)
        bse = bae = 0.0
        n = 0
        for batch_x, batch_y, _, _ in loader:
            X, T, w, base = _batch(model, batch_x, batch_y, H, Lp, device)
            s = w.sqrt()[:, :, None, None]  # sigma
            bse += float(((s * (base - T)) ** 2).sum())
            bae += float((s * (base - T)).abs().sum())
            for k0 in range(0, K, 8):
                P = torch.einsum('bclf,kclfp->kbclp', X, Wst[k0:k0 + 8])
                E = s[None] * (P - T[None])
                se[k0:k0 + 8] += (E ** 2).flatten(1).sum(1)
                ae[k0:k0 + 8] += E.abs().flatten(1).sum(1)
            n += T.numel()
        return (se / n).cpu().numpy(), (ae / n).cpu().numpy(), bse / n, bae / n

    v_mse, v_mae, vb_mse, vb_mae = evaluate('val')
    t_mse, t_mae, tb_mse, tb_mae = evaluate('test')

    best = int(np.argmin(v_mse))
    per_variant = {}
    for fs in FEATS:
        for g in GROUPS:
            ks = [k for k, c in enumerate(combos) if c[0] == fs and c[1] == g]
            kb = min(ks, key=lambda k: v_mse[k])
            per_variant['%s|%s' % (fs, g)] = dict(lam=combos[kb][2], val_mse=float(v_mse[kb]),
                                                   test_mse=float(t_mse[kb]), test_mae=float(t_mae[kb]))
    out = dict(setting=setting, pred_len=H,
               base=dict(val_mse=vb_mse, test_mse=tb_mse, test_mae=tb_mae),
               chosen=dict(feats=combos[best][0], group=combos[best][1], lam=combos[best][2],
                           val_mse=float(v_mse[best]), test_mse=float(t_mse[best]), test_mae=float(t_mae[best])),
               variants=per_variant)
    folder = './results/' + setting + '/'
    os.makedirs(folder, exist_ok=True)
    with open(folder + 'refit.json', 'w') as f:
        json.dump(out, f, indent=2)
    c = out['chosen']
    print('[refit] base   test MSE %.4f MAE %.4f (val %.4f)' % (tb_mse, tb_mae, vb_mse))
    print('[refit] chosen %s/%s lam=%g  test MSE %.4f (%+.2f%%) MAE %.4f (%+.2f%%)  (val %.4f)' % (
        c['feats'], c['group'], c['lam'], c['test_mse'], (c['test_mse'] / tb_mse - 1) * 100,
        c['test_mae'], (c['test_mae'] / tb_mae - 1) * 100, c['val_mse']))
    for k, v in per_variant.items():
        print('[refit]   %-16s lam=%-7g val %.4f  test MSE %.4f (%+.2f%%)  MAE %.4f (%+.2f%%)' % (
            k, v['lam'], v['val_mse'], v['test_mse'], (v['test_mse'] / tb_mse - 1) * 100,
            v['test_mae'], (v['test_mae'] / tb_mae - 1) * 100))
    return out
