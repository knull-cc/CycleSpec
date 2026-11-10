"""PhaseFormer + BDR: ReNF's Boosted Direct Output as a backbone-agnostic plug-in, in data space.

The backbone (here PhaseFormer, unchanged) gives a forecast Y0. The refiner then runs ReNF's BDO
recursion on top of it, with the concatenation done in DATA space as in ReNF:

    stage k:  Y_k = Y0[:h_k] + MLP_k([x_recent, Y_(k-1)])        Y_0 := Y0 (full horizon)
    output  = Y_n  (h_n = H)

* every stage has its own small MLP (Linear -> GELU -> Linear), so each stage learns its own
  combination function f(x) + phi(previous forecast), which is BDO's "implicit combination";
* the last layer of every MLP is zero-initialised: at step 0 the model IS the backbone;
* everything is computed per channel on instance-normalised values (input mean / std), so the
  refiner only needs the backbone's input and output -> works for any forecaster;
* no dropout, and its parameters are initialised from a forked RNG: the global random stream
  (backbone init, data order, backbone dropout) is identical to a plain PhaseFormer run with the
  same seed, so a 1-seed comparison against PhaseFormer isolates the plug-in's effect;
* stage losses: --bdo_w_mode last (default here) | equal | inv, via exp._train_loss.

args: --bdr_stages (3) --bdr_hidden (64) --bdr_ctx (96, recent input steps; 0 = whole input)
"""
import math

import torch
import torch.nn as nn

from models.PhaseFormer import Model as PhaseFormerModel


def prefix_lengths(pred_len, n_stages):
    n = max(1, int(n_stages))
    return [min(pred_len, max(1, math.ceil(pred_len * (k + 1) / n))) for k in range(n)]


class BDORefiner(nn.Module):
    def __init__(self, seq_len, pred_len, n_stages=3, hidden=64, ctx=96, eps=1e-5):
        super().__init__()
        self.ctx = min(seq_len, ctx) if ctx and ctx > 0 else seq_len
        self.lens = prefix_lengths(pred_len, n_stages)
        self.eps = eps
        self.stages = nn.ModuleList()
        prev = pred_len  # stage 1 sees the full backbone forecast
        for h in self.lens:
            last = nn.Linear(hidden, h)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.stages.append(nn.Sequential(nn.Linear(self.ctx + prev, hidden), nn.GELU(), last))
            prev = h

    def forward(self, x, y0):
        """x (B,L,C) backbone input, y0 (B,H,C) backbone forecast -> list of stage forecasts (B,h_k,C)."""
        mu = x.mean(1, keepdim=True)
        sd = (x.var(1, keepdim=True, unbiased=False) + self.eps).sqrt()
        xn = ((x[:, -self.ctx:] - mu) / sd).permute(0, 2, 1)  # (B,C,ctx)
        yn = ((y0 - mu) / sd).permute(0, 2, 1)  # (B,C,H)
        outs, prev = [], yn
        for h, mlp in zip(self.lens, self.stages):
            yk = yn[..., :h] + mlp(torch.cat([xn, prev], dim=-1))
            outs.append(yk.permute(0, 2, 1) * sd + mu)
            prev = yk
        return outs


class Model(PhaseFormerModel):
    def __init__(self, configs):
        super().__init__(configs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(getattr(configs, 'fix_seed', 2021)) + 7919)
            self.refiner = BDORefiner(configs.seq_len, configs.pred_len,
                                      n_stages=int(getattr(configs, 'bdr_stages', 3)),
                                      hidden=int(getattr(configs, 'bdr_hidden', 64)),
                                      ctx=int(getattr(configs, 'bdr_ctx', 96)))
        self.stage_outputs = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        y0 = super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
        outs = self.refiner(x_enc, y0)
        self.stage_outputs = outs if self.training else None
        return outs[-1]
