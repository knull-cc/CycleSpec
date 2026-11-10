"""Over-parameterised linear head that collapses to a single nn.Linear for inference.

Every long-term forecaster (DLinear, PatchTST, iTransformer, PhaseFormer, ...) ends in an affine
map  y = W z + b.  RepLinear keeps exactly that function class but trains it through a deeper,
purely linear parameterisation (no activation in between), so only the optimisation dynamics
change.  After training, `merged()` returns the equivalent nn.Linear(in_dim, out_dim): same
parameter count and same inference cost as the original head.

modes
    linear   plain nn.Linear (reference, identical to the original head)
    chain    Linear(in, w) -> Linear(w, w) x (depth-2) -> Linear(w, out)      (ExpandNets-style)
    cascade  stage j: Linear(in + sum_{i<j} p_i, p_j) on [z, y_1, ..., y_{j-1}], output = y_n
             (stage sizes p_1 <= ... <= p_n = out, prefixes of the output vector; this is the
             'E' arm of the BDO experiment, i.e. BDO trained with the last-stage loss only)
"""
import math

import torch
import torch.nn as nn


def cascade_sizes(out_dim, n_stages):
    n = max(1, int(n_stages))
    return [min(out_dim, max(1, math.ceil(out_dim * (j + 1) / n))) for j in range(n)]


class RepLinear(nn.Module):
    def __init__(self, in_dim, out_dim, mode='chain', depth=2, width=0, stages=3):
        super().__init__()
        self.in_dim, self.out_dim, self.mode = int(in_dim), int(out_dim), mode
        if mode == 'linear':
            self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        elif mode == 'chain':
            depth = max(1, int(depth))
            w = int(width) if int(width) > 0 else 4 * max(in_dim, out_dim)
            dims = [in_dim] + [w] * (depth - 1) + [out_dim]
            # only the last layer carries a bias: the composed bias is then exactly that bias
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=(i == depth - 1))
                                         for i in range(depth)])
        elif mode == 'cascade':
            self.sizes = cascade_sizes(out_dim, stages)
            layers, n_prev = [], 0
            for p in self.sizes:
                layers.append(nn.Linear(in_dim + n_prev, p))
                n_prev += p
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError('rep_mode must be linear | chain | cascade, got %s' % mode)

    def forward(self, z):
        if self.mode == 'cascade':
            ys = []
            for layer in self.layers:
                ys.append(layer(z if not ys else torch.cat([z] + ys, dim=-1)))
            return ys[-1]
        for layer in self.layers:
            z = layer(z)
        return z

    @torch.no_grad()
    def merged(self):
        """Equivalent nn.Linear(in_dim, out_dim), found by probing the affine map."""
        p = next(self.parameters())
        eye = torch.eye(self.in_dim, device=p.device, dtype=p.dtype)
        zero = torch.zeros(1, self.in_dim, device=p.device, dtype=p.dtype)
        b = self.forward(zero)[0]                       # f(0) = b
        W = (self.forward(eye) - b).t().contiguous()     # f(e_i) - b = W[:, i]
        lin = nn.Linear(self.in_dim, self.out_dim).to(device=p.device, dtype=p.dtype)
        lin.weight.copy_(W)
        lin.bias.copy_(b)
        return lin
