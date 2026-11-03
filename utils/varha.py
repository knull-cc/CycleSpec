"""VarHA: TT-HA (Cui et al., CVPR 2026) style test-time heterogeneous adaptation,
moved to a frozen PhaseFormer as a lightweight plug-in.

TT-HA -> PhaseFormer mapping
  anatomical part p          -> variable group p (spectral clustering of |corr| on TRAIN)
  part parameters Theta_p    -> a per-group copy of PhaseFormer's last layer
                                (model.predictor: Linear(D -> P_out), 20 numbers at the
                                ETTh1 H=96 config), initialised from the source weights
  shared parameters          -> the rest of PhaseFormer, frozen
  peak -> reset Theta_p      -> Theta_p <- source weights, i.e. back to the frozen model
  no peak -> adapt Theta_p   -> one SGD step on ground truth that has already arrived

Per test batch of B consecutive windows (time order, never shuffled):
  1. drift score per group: W1 between 1-D Gaussians of the group's latent statistics
     (mean / std of Z over phases, as TT-HA's instance-norm statistics) and their EMA
  2. peak: z-score of the score against its last `win` values >= tau  (tau=inf: off)
  3. peak groups reset, the others take one SGD step on the partially observed labels:
     step h of an earlier window j is used only if t_j + h < t_0, where t_0 is the first
     target step of the batch being predicted (so it lies inside that batch's input)
  4. predict with the adapted head; input series and backbone are never modified

The head is linear in the cached latent Z, so the gradient is closed form: the whole
online loop is numpy on cached backbone features (no autograd, no extra backbone pass).

No leakage
  - labels: only steps strictly before the current prediction origin (see 3.)
  - grouping: train split only
  - hyper-parameters: chosen on the validation stream only (select_on_val);
    a config is accepted only if it lowers BOTH val MSE and val MAE, else identity
  - test is run once, as one continuous stream val -> test (all val labels end
    before the first test target step, so the warm-up on val is causal)
"""

from dataclasses import dataclass, asdict, replace
from itertools import product
from typing import Dict, Optional

import numpy as np
from scipy.special import erf


# --------------------------------------------------------------------------- grouping
def group_variables(train_values: np.ndarray, n_groups: int, seed: int = 0) -> np.ndarray:
    """Variable groups from the TRAIN split only: spectral clustering of |corr|.
    train_values (T, C) -> integer labels (C,). n_groups=1: one group; >=C: one per variable."""
    C = train_values.shape[1]
    if n_groups <= 1:
        return np.zeros(C, dtype=int)
    if n_groups >= C:
        return np.arange(C)
    A = np.nan_to_num(np.abs(np.corrcoef(train_values.T)), nan=0.0)
    np.fill_diagonal(A, 0.0)
    d = A.sum(1) + 1e-12
    _, vec = np.linalg.eigh(np.eye(C) - A / np.sqrt(d[:, None] * d[None, :]))
    U = vec[:, :n_groups]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return _kmeans(U, n_groups, seed)


def random_groups(C: int, n_groups: int, seed: int) -> np.ndarray:
    """Random partition into n_groups balanced groups (ablation B8)."""
    labels = np.arange(C) % max(1, min(n_groups, C))
    np.random.RandomState(seed).shuffle(labels)
    return labels


def _kmeans(X, k, seed, iters=100, restarts=10):
    rng = np.random.RandomState(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        cent = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
            new = np.stack([X[lab == j].mean(0) if np.any(lab == j) else cent[j] for j in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        inertia = ((X - cent[lab]) ** 2).sum()
        if inertia < best_inertia and len(np.unique(lab)) == k:
            best, best_inertia = lab, inertia
    if best is None:
        best = np.arange(len(X)) % k
    _, first = np.unique(best, return_index=True)  # relabel by first appearance
    remap = np.empty(k, dtype=int)
    remap[np.argsort(first)] = np.arange(k)
    return remap[best]


# --------------------------------------------------------------------------- drift score
def gaussian_w1(mu1, sd1, mu2, sd2):
    """W1 between N(mu1, sd1^2) and N(mu2, sd2^2) = E|delta + s Z| (folded-normal mean),
    delta = mu1 - mu2, s = sd1 - sd2."""
    d = mu1 - mu2
    s = np.abs(sd1 - sd2)
    safe = np.maximum(s, 1e-12)
    w = s * np.sqrt(2.0 / np.pi) * np.exp(-d ** 2 / (2 * safe ** 2)) + d * erf(d / (np.sqrt(2.0) * safe))
    return np.where(s < 1e-12, np.abs(d), w)


# --------------------------------------------------------------------------- PhaseFormer head
def head_forward(z, W, b, labels, period_len, pred_len):
    """PhaseFormer's predictor + reassembly, per variable through its group's weights.
    z (n, C, L, D) latent; W (G, P_out, D); b (G, P_out) -> (n, H, C) in RevIN space."""
    yph = np.einsum('ncld,ckd->nclk', z, W[labels]) + b[labels][None, :, None, :]  # (n, C, L, P_out)
    n, C = z.shape[:2]
    return yph.transpose(0, 1, 3, 2).reshape(n, C, -1)[:, :, :pred_len].transpose(0, 2, 1)


def _to_phase(e, period_len, p_out):
    """(n, H, C) -> (n, C, L, P_out), zero-padding the last period (inverse of the reassembly)."""
    n, H, C = e.shape
    full = np.zeros((n, C, p_out * period_len))
    full[:, :, :H] = e.transpose(0, 2, 1)
    return full.reshape(n, C, p_out, period_len).transpose(0, 1, 3, 2)


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class VarHAConfig:
    n_groups: int = 1           # P (1 = uniform adaptation)
    lr: float = 1e-3            # SGD step on the RevIN-space loss
    batch: int = 24             # windows per online step
    delta: float = float('inf')  # Huber delta of the adaptation loss (inf = MSE)
    tau: float = float('inf')   # peak threshold on the z-score (inf = reset off)
    win: int = 36               # z-score history (TT-HA default)
    ema: float = 0.95           # weight of the NEW batch statistics in the EMA (TT-HA)
    random_groups: bool = False
    seed: int = 0

    def tag(self):
        f = lambda v: 'off' if not np.isfinite(v) else '{:g}'.format(v)
        return 'P{}_lr{:g}_B{}_huber{}_tau{}{}'.format(self.n_groups, self.lr, self.batch, f(self.delta),
                                                      f(self.tau), '_rand' if self.random_groups else '')


# --------------------------------------------------------------------------- online learner
class VarHA:
    def __init__(self, cfg: VarHAConfig, labels, W0, b0, period_len, pred_len):
        self.cfg, self.labels = cfg, np.asarray(labels, dtype=int)
        self.G = int(self.labels.max()) + 1
        self.W0, self.b0 = np.asarray(W0, np.float64), np.asarray(b0, np.float64)
        self.W = np.repeat(self.W0[None], self.G, 0)
        self.b = np.repeat(self.b0[None], self.G, 0)
        self.period_len, self.H, self.p_out = period_len, pred_len, self.W0.shape[0]
        self.ema_mu = self.ema_sd = None
        self.hist = [[] for _ in range(self.G)]
        self.n_resets = np.zeros(self.G, dtype=int)
        self.n_steps = 0

    def predict(self, z):
        return head_forward(z, self.W, self.b, self.labels, self.period_len, self.H)

    def peaks(self, z):
        """Drift score and peak test from the current batch's latent (input only)."""
        cfg, G = self.cfg, self.G
        fmu, fsd = z.mean(2), z.std(2)                                    # (n, C, D)
        mu = np.stack([fmu[:, self.labels == g].mean((0, 1)) for g in range(G)])
        sd = np.stack([fsd[:, self.labels == g].mean((0, 1)) for g in range(G)])
        pk = np.zeros(G, dtype=bool)
        if self.ema_mu is None:
            self.ema_mu, self.ema_sd = mu, sd
            return pk
        omega = gaussian_w1(self.ema_mu, self.ema_sd, mu, sd).mean(1)    # (G,)
        for g in range(G):
            h = self.hist[g]
            if np.isfinite(cfg.tau) and len(h) >= max(8, cfg.win // 3):
                ref = np.asarray(h[-cfg.win:])
                pk[g] = (omega[g] - ref.mean()) / (ref.std() + 1e-8) >= cfg.tau
            h.append(float(omega[g]))
        self.ema_mu = (1 - cfg.ema) * self.ema_mu + cfg.ema * mu
        self.ema_sd = (1 - cfg.ema) * self.ema_sd + cfg.ema * sd
        return pk

    def reset(self, pk):
        self.W[pk], self.b[pk] = self.W0, self.b0
        self.n_resets += pk

    def step(self, z, yt_n, mask, frozen):
        """One SGD step on the masked (Huber) loss. z (m, C, L, D); yt_n (m, H, C) truth in
        RevIN space (NaN where unobserved); mask (m, H) observed steps."""
        e = self.predict(z) - np.nan_to_num(yt_n)
        e = np.clip(e, -self.cfg.delta, self.cfg.delta) * mask[:, :, None]   # Huber gradient
        e_ph = _to_phase(e, self.period_len, self.p_out)                     # (m, C, L, P_out)
        for g in range(self.G):
            if frozen[g]:
                continue
            c = self.labels == g
            cnt = max(mask.sum() * c.sum(), 1)
            self.W[g] -= self.cfg.lr * 2.0 * self.p_out * np.einsum('mclk,mcld->kd', e_ph[:, c], z[:, c]) / cnt
            self.b[g] -= self.cfg.lr * 2.0 * self.p_out * e_ph[:, c].sum((0, 1, 2)) / cnt
        self.n_steps += 1


# --------------------------------------------------------------------------- stream runner
def run_stream(cfg: VarHAConfig, labels, st: Dict[str, np.ndarray], head: Dict):
    """Run VarHA over a time-ordered stream.
    st:   z (n, C, L, D) latent fed to the predictor; mu, sd (n, C) RevIN stats of the input;
          y (n, H, C) truth in scaler space; t (n,) absolute index of each window's first target step.
    head: W0 (P_out, D), b0 (P_out,), period_len, pred_len.
    Returns predictions (n, H, C) in scaler space and the learner."""
    z, mu, sd, y, t = st['z'], st['mu'], st['sd'], st['y'], st['t']
    H = head['pred_len']
    ln = VarHA(cfg, labels, head['W0'], head['b0'], head['period_len'], H)
    out = np.empty(y.shape, dtype=np.float64)
    for s in range(0, len(t), cfg.batch):
        e, t0 = min(len(t), s + cfg.batch), t[s]
        pk = ln.peaks(z[s:e])
        ln.reset(pk)
        lo, hi = int(np.searchsorted(t, t0 - H, 'left')), int(np.searchsorted(t, t0, 'left'))
        if hi > lo:  # windows started before t0 whose target is not complete: t0 - H <= t_j < t0
            j = slice(lo, hi)
            mask = (t[j][:, None] + np.arange(H)[None]) < t0             # step observed before t0
            yt = np.where(mask[:, :, None], y[j], np.nan)                 # future steps never read
            ln.step(z[j], (yt - mu[j][:, None, :]) / sd[j][:, None, :], mask, pk)
        out[s:e] = ln.predict(z[s:e]) * sd[s:e][:, None, :] + mu[s:e][:, None, :]
    return out, ln


def metrics(pred, true):
    return float(np.mean((pred - true) ** 2)), float(np.mean(np.abs(pred - true)))


def base_pred(st, head):
    C = st['z'].shape[1]
    yn = head_forward(st['z'], np.asarray(head['W0'])[None], np.asarray(head['b0'])[None],
                      np.zeros(C, dtype=int), head['period_len'], head['pred_len'])
    return yn * st['sd'][:, None, :] + st['mu'][:, None, :]


# --------------------------------------------------------------------------- protocol
DEFAULT_GRID = dict(n_groups=[1, 3], lr=[1e-3, 3e-3, 1e-2], delta=[float('inf'), 1.0], tau=[float('inf'), 3.0])


def _labels(cfg, train_values):
    C = train_values.shape[1]
    if cfg.random_groups:
        return random_groups(C, cfg.n_groups, cfg.seed)
    return group_variables(train_values, cfg.n_groups, cfg.seed)


def select_on_val(val, head, train_values, base_cfg=VarHAConfig(), grid=None, log=print):
    """Grid search on the validation stream only (cold start from the source head).
    Eligible = lowers both val MSE and val MAE; best = lowest MSE/MSE0 + MAE/MAE0.
    Returns (cfg or None, rows)."""
    grid = grid or DEFAULT_GRID
    C = train_values.shape[1]
    b_mse, b_mae = metrics(base_pred(val, head), val['y'])
    cfgs = list(dict.fromkeys(
        replace(base_cfg, n_groups=min(int(P), C), lr=float(lr), delta=float(dl), tau=float(tau))
        for P, lr, dl, tau in product(grid['n_groups'], grid['lr'], grid['delta'], grid['tau'])))
    rows, best, best_s = [], None, np.inf
    for cfg in cfgs:
        pred, ln = run_stream(cfg, _labels(cfg, train_values), val, head)
        mse, mae = metrics(pred, val['y'])
        s = mse / b_mse + mae / b_mae
        rows.append(dict(cfg=cfg.tag(), mse=mse, mae=mae, score=s, resets=int(ln.n_resets.sum())))
        if mse < b_mse and mae < b_mae and s < best_s:
            best, best_s = cfg, s
    log('[VarHA] val base               mse={:.5f} mae={:.5f}'.format(b_mse, b_mae))
    for r in sorted(rows, key=lambda r: r['score'])[:6]:
        log('[VarHA] val {:<38} mse={:.5f} mae={:.5f} resets={}'.format(r['cfg'], r['mse'], r['mae'], r['resets']))
    log('[VarHA] selected on val: {}'.format(best.tag() if best else 'none -> identity (plug-in off)'))
    return best, rows


def concat_streams(a, b):
    return {k: np.concatenate([a[k], b[k]], 0) for k in a}


def run_test(cfg, val, test, head, train_values):
    """One continuous stream val -> test with a fixed config; returns test predictions."""
    pred, ln = run_stream(cfg, _labels(cfg, train_values), concat_streams(val, test), head)
    return pred[len(val['t']):], ln


def run_protocol(val, test, head, train_values, base_cfg=VarHAConfig(), grid=None, log=print):
    """Select on val, run test once, plus ablations at the selected step size:
    B1 uniform (P=1), B3 no reset, the full TT-HA recipe (P>=3 groups + reset, tau=3 if
    val turned reset off) and B8 = that recipe with random groups. The ablations are
    reported for analysis only; B2 is the val-selected number."""
    res = {'B0_PhaseFormer': metrics(base_pred(test, head), test['y'])}
    best, rows = select_on_val(val, head, train_values, base_cfg, grid, log)
    out = dict(results=res, selected=None, val_rows=rows)
    if best is None:
        res['B2_VarHA'] = res['B0_PhaseFormer']
        return out
    pred, ln = run_test(best, val, test, head, train_values)
    res['B2_VarHA'] = metrics(pred, test['y'])
    out.update(selected=asdict(best), selected_tag=best.tag(), pred=pred,
               info=dict(groups=_labels(best, train_values).tolist(), resets=ln.n_resets.tolist(),
                         steps=ln.n_steps, trainable=int(ln.W.size + ln.b.size)))
    # the full TT-HA recipe (groups + reset) at the selected step size, whether or not val picked it
    tt = replace(best, n_groups=max(best.n_groups, min(3, train_values.shape[1])),
                 tau=best.tau if np.isfinite(best.tau) else 3.0, random_groups=False)
    abl = {'B1_uniform_P1': replace(best, n_groups=1, random_groups=False),
           'B3_no_reset': replace(best, tau=float('inf')),
           'TTHA_groups+reset': tt,
           'B8_random_groups': replace(tt, random_groups=True)}
    for name, cfg in abl.items():
        res[name] = res['B2_VarHA'] if cfg == best else metrics(run_test(cfg, val, test, head, train_values)[0], test['y'])
    b = res['B0_PhaseFormer']
    for k, (mse, mae) in res.items():
        log('[VarHA] test {:<17} mse={:.5f} ({:+.2f}%)  mae={:.5f} ({:+.2f}%)'.format(
            k, mse, (mse / b[0] - 1) * 100, mae, (mae / b[1] - 1) * 100))
    return out
