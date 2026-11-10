"""PhaseFormer + BDO output head (cascaded, prefix-supervised direct output).

Idea taken from ReNF (ICML 2026, arXiv 2509.25914): instead of one head that emits the whole
horizon at once, emit it in n stages. Stage j sees the shared representation plus ALL earlier
stage predictions and emits a longer prefix of the horizon; every stage is supervised on its own
prefix. The final stage is the model output.

Adaptation to PhaseFormer: the cascade runs in *phase space*. PhaseFormer's last step is
Z (B,C,L,D) -> Linear(D, P_out) giving, for each of the L=period_len phases, P_out future periods.
Here stage j is Linear(D + sum_{i<j} P_i, P_j) on [Z, y_1, ..., y_{j-1}] with P_j non-decreasing
and P_n = P_out, so a stage covers the first P_j periods of the horizon (a time prefix).

Everything before the head (RevIN, embedding, routing layers) is PhaseFormer unchanged, so the
same wrapper works on any backbone that exposes a pre-head representation.
With bdo_stages=1 the model is exactly PhaseFormer (same maths, same parameter count).

Training: forward() stores the denormalised prefix predictions in `self.stage_outputs`
(only while self.training); the experiment loop turns them into the multi-stage loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PhaseFormer import Model as PhaseFormerModel
from utils.bdo_utils import bdo_stage_periods


class Model(PhaseFormerModel):
    def __init__(self, configs):
        super().__init__(configs)
        if getattr(configs, "predictor_use_mlp", False):
            raise NotImplementedError("PhaseFormerBDO supports the default linear predictor only")

        self.bdo_stages = int(getattr(configs, "bdo_stages", 3))
        self.stage_periods = bdo_stage_periods(self.num_periods_output, self.bdo_stages)

        # replace PhaseFormer's single predictor by the cascade of heads
        del self.predictor
        drop = float(getattr(configs, "predictor_dropout", 0.0))
        self.head_dropout = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        heads, n_prev = [], 0
        for p_j in self.stage_periods:
            heads.append(nn.Linear(self.latent_dim + n_prev, p_j))
            n_prev += p_j
        self.heads = nn.ModuleList(heads)

        self.stage_outputs = None

    # steps 1-8 of PhaseFormer.forward, unchanged, returning the latent before the predictor
    def _encode(self, x_enc):
        if self.use_revin:
            x_in, stats = self.revin.normalize(x_enc)
        else:
            x_in, stats = x_enc.float(), None
        x = x_in.permute(0, 2, 1)  # (B, C, L_in)
        B, C, _ = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)
        phase_series = self._to_phase_series(x_periods)  # (B, C, L, P_in)

        Z = self.embedding(phase_series)
        phase_series_cur = phase_series
        for layer_index, unit in enumerate(self.routing_layers):
            Z, y_phase_steps_p_in = unit(phase_series_cur, Z)
            if layer_index < len(self.routing_layers) - 1:
                phase_series_cur = y_phase_steps_p_in
        return Z, stats, B, C

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        Z, stats, B, C = self._encode(x_enc)
        Zd = self.head_dropout(Z)

        ys = []  # list of (B, C, L, P_j), normalised space
        for head in self.heads:
            inp = Zd if not ys else torch.cat([Zd] + ys, dim=-1)
            ys.append(head(inp))

        seqs = []
        for y in ys:
            seq = self._from_phase_steps_to_periods(y).reshape(B, C, -1)[..., : self.pred_len]
            seq = seq.permute(0, 2, 1)  # (B, len_j, C)
            if self.use_revin:
                seq = self.revin.denormalize(seq, stats)
            seqs.append(seq)

        self.stage_outputs = seqs if self.training else None
        return seqs[-1]
