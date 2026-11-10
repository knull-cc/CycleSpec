"""CycleSpec: a backbone-agnostic, Frame2Freq-style spectral adapter.

One module, one contract:  CycleSpecAdapter(x) with x of shape (N, T, F) -> (N, T, F)
    N : everything the adapter treats independently (batch x variables x ...)
    T : an ordered axis the adapter mixes along (STFT + depthwise conv)
    F : features at each position
    out = x + FC_up([freq-branch | time-branch](GELU(FC_down(x))))   with FC_up = 0 at init

    windows = [T, T//2, T//4], hop = window // 4        (Frame2Freq's K=3 rule, no constants)
    periodic axis -> circular padding, otherwise reflect padding
    series are processed in chunks (CUDA launch limits on very large N)

A host model only decides WHERE to call it ("ports") and how to view the tensor found there as
(N, T, F). Ports are attached with forward hooks, so the host's code is not modified.

PhaseFormer ports (select with --cs_port input | block | both | output | io):
    input : the RevIN-normalised input as cycles, before the phase embedding
            tensor (B, C, 24 phases, 30 cycles) -> (B*C, T=30 cycles, F=24), non-periodic
    block : after the cross-phase routing attention (+ residual + LayerNorm) of every routing layer,
            before its MLP (the "after MHSA in every block" position of Frame2Freq)
            tensor (B*C, 24 phases, D) -> T=24 phases (periodic: hour 23 is next to hour 0), F=D
    output: on the forecast. The past cycles (RevIN-normalised input) and the predicted future cycles
            are joined along the cycle axis, (B*C, 30 + P_out, 24); the adapter refines this sequence
            and only the future cycles are kept. It sees how each day's shape and level evolve
            across the history and into the forecast (works for any forecaster that outputs a series).
    both = input + block, io = input + output

--cs_band restricts the correction Delta to a band of the axis (applied to Delta with an rFFT mask, so
it holds whatever the branches and biases do):
    all  : every frequency (original behaviour)
    nodc : everything except bin 0, so the mean along the axis is never changed
    mid  : only periods of 3.75 to 10 positions along the axis (bins 3..8 for T=30 cycles)

--cs_arch dense replaces the STFT branch by a dense Linear(T -> T) on the same bottleneck
(capacity control: does the spectral structure matter, or only the extra parameters?).
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PhaseFormer import Model as PhaseFormer


# ============================================================================ core module
class SpectralScale(nn.Module):
    """STFT along T -> depthwise conv over frequency bins (real/imag share weights) -> iSTFT."""

    def __init__(self, channels: int, win: int, kernel: int = 3, periodic: bool = False):
        super().__init__()
        self.win = int(win)
        self.hop = max(1, self.win // 4)
        self.pad_mode = "circular" if periodic else "reflect"
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)

    def _stft(self, x):  # (M, T) -> (M, Fq, Fr)
        return torch.stft(x, n_fft=self.win, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=True, pad_mode=self.pad_mode, normalized=False, onesided=True,
                          return_complex=True)

    def _istft(self, spec, length):
        return torch.istft(spec, n_fft=self.win, hop_length=self.hop, win_length=self.win, window=self.window,
                           center=True, normalized=False, onesided=True, length=length)

    def forward(self, u, apply_conv: bool = True):  # (N, T, c) -> (N, T, c)
        N, T, c = u.shape
        spec = self._stft(u.permute(0, 2, 1).reshape(N * c, T))
        if apply_conv:
            Fq, Fr = spec.shape[-2], spec.shape[-1]
            spec = spec.reshape(N, c, Fq, Fr).permute(0, 3, 1, 2).reshape(N * Fr, c, Fq)
            spec = torch.complex(self.conv(spec.real.contiguous()), self.conv(spec.imag.contiguous()))
            spec = spec.reshape(N, Fr, c, Fq).permute(0, 2, 3, 1).reshape(N * c, Fq, Fr)
        return self._istft(spec, T).reshape(N, c, T).permute(0, 2, 1)


class DenseMix(nn.Module):
    """Capacity control for the frequency branch: a dense Linear(T -> T) along the axis."""

    def __init__(self, T: int):
        super().__init__()
        self.lin = nn.Linear(T, T)

    def forward(self, u):  # (N, T, c)
        return self.lin(u.permute(0, 2, 1)).permute(0, 2, 1)


def band_mask(T: int, band: str):
    """Boolean mask over rFFT bins 0..T//2 of an axis of length T."""
    k = torch.arange(T // 2 + 1, dtype=torch.float32)
    if band == "all":
        return torch.ones_like(k, dtype=torch.bool)
    if band == "nodc":
        return k > 0
    if band == "mid":  # periods between 3.75 and 10 positions: T/10 <= k <= T/3.75
        return (k >= T / 10.0 - 1e-6) & (k <= T / 3.75 + 1e-6) & (k > 0)
    raise ValueError(band)


def default_windows(T: int):
    return sorted({max(2, T), max(2, T // 2), max(2, T // 4)}, reverse=True)


class CycleSpecAdapter(nn.Module):
    """Residual adapter on an ordered axis. Input/output: (N, T, F). Identity at initialisation."""

    def __init__(self, feat_dim: int, axis_len: int, hidden: int = 8, windows: Optional[Sequence[int]] = None,
                 kernel: int = 3, branches: str = "both", act: str = "gelu", arch: str = "spectral",
                 periodic: bool = False, chunk: int = 8192, band: str = "all"):
        super().__init__()
        assert branches in ("both", "freq", "time"), branches
        assert act in ("gelu", "none"), act
        assert arch in ("spectral", "dense"), arch
        self.feat_dim, self.axis_len, self.hidden = feat_dim, axis_len, hidden
        self.branches, self.act, self.arch, self.periodic = branches, act, arch, periodic
        self.chunk = int(chunk)
        self.band = band
        self.register_buffer("bmask", band_mask(axis_len, band), persistent=False)
        if branches == "both":
            self.c_freq = hidden // 2
            self.c_time = hidden - self.c_freq
        elif branches == "freq":
            self.c_freq, self.c_time = hidden, 0
        else:
            self.c_freq, self.c_time = 0, hidden
        self.windows = [min(int(w), axis_len) for w in (windows or default_windows(axis_len))]

        self.fc_down = nn.Linear(feat_dim, hidden)
        if self.c_freq > 0 and arch == "spectral":
            self.freq_scales = nn.ModuleList([SpectralScale(self.c_freq, w, kernel, periodic) for w in self.windows])
        else:
            self.freq_scales = nn.ModuleList()
        self.dense = DenseMix(axis_len) if (self.c_freq > 0 and arch == "dense") else None
        self.time_conv = (nn.Conv1d(self.c_time, self.c_time, kernel, padding=kernel // 2, groups=self.c_time,
                                    padding_mode="circular" if periodic else "zeros")
                          if self.c_time > 0 else None)
        self.fc_up = nn.Linear(hidden, feat_dim)
        nn.init.zeros_(self.fc_up.weight)
        nn.init.zeros_(self.fc_up.bias)

    def delta(self, x):  # (N, T, F) -> (N, T, F)
        if self.chunk > 0 and x.shape[0] > self.chunk:
            return torch.cat([self._delta(x[i:i + self.chunk]) for i in range(0, x.shape[0], self.chunk)], dim=0)
        return self._delta(x)

    def _delta(self, x):
        u = self.fc_down(x)
        if self.act == "gelu":
            u = F.gelu(u)
        parts = []
        if self.c_freq > 0:
            u_f = u[..., : self.c_freq]
            if self.dense is not None:
                parts.append(self.dense(u_f))
            else:
                parts.append(torch.stack([s(u_f) for s in self.freq_scales], dim=0).mean(dim=0))
        if self.c_time > 0:
            u_t = u[..., self.c_freq:]
            parts.append(self.time_conv(u_t.permute(0, 2, 1)).permute(0, 2, 1))
        d = self.fc_up(torch.cat(parts, dim=-1))
        if self.band != "all":  # keep only the chosen band of Delta along the axis
            T = d.shape[1]
            D = torch.fft.rfft(d, dim=1) * self.bmask.to(d.dtype)[None, :, None]
            d = torch.fft.irfft(D, n=T, dim=1)
        return d

    def forward(self, x):
        return x + self.delta(x)


# ============================================================================ ports (hooks)
class _Probe:
    """Collects ||delta|| / ||x|| for each adapter call while active (diagnostics only)."""

    def __init__(self):
        self.active, self.values = False, []


def _apply(adapter, x_ntf, probe):
    d = adapter.delta(x_ntf)
    if probe.active:
        probe.values.append((d.norm() / (x_ntf.norm() + 1e-12)).item())
    return x_ntf + d


def attach_input_port(module, adapter, to_ntf, from_ntf, probe):
    """Wrap the first positional INPUT of `module`: x -> from_ntf(adapter(to_ntf(x)), x)."""
    def pre_hook(mod, args):
        x = args[0]
        y = from_ntf(_apply(adapter, to_ntf(x), probe), x)
        return (y,) + tuple(args[1:])
    return module.register_forward_pre_hook(pre_hook)


def attach_output_port(module, adapter, to_ntf, from_ntf, probe):
    """Wrap the OUTPUT of `module`: y -> from_ntf(adapter(to_ntf(y)), y)."""
    def hook(mod, args, out):
        return from_ntf(_apply(adapter, to_ntf(out), probe), out)
    return module.register_forward_hook(hook)


# ============================================================================ PhaseFormer + CycleSpec
class Model(PhaseFormer):
    """PhaseFormer with CycleSpec adapters attached at the chosen port(s)."""

    PORTS = ("input", "block", "both", "output", "io")

    def __init__(self, configs):
        super().__init__(configs)  # identical backbone, identical RNG consumption
        self.use_cyclespec = True
        self.freeze_backbone = False
        self.cs_port = getattr(configs, "cs_port", "input")
        assert self.cs_port in self.PORTS, self.cs_port
        self._probe = _Probe()

        common = dict(kernel=int(getattr(configs, "cs_kernel", 3)), branches=getattr(configs, "cs_branches", "both"),
                      act=getattr(configs, "cs_act", "gelu"), arch=getattr(configs, "cs_arch", "spectral"),
                      chunk=int(getattr(configs, "cs_chunk", 8192)), band=getattr(configs, "cs_band", "all"))
        windows = getattr(configs, "cs_windows", "auto")
        windows = None if windows in (None, "", "auto") else [int(w) for w in str(windows).split(",") if w.strip()]
        block_hidden = int(getattr(configs, "cs_block_hidden", 0)) or max(2, min(8, self.latent_dim))

        # adapters are built after the backbone and under a forked CPU RNG: the backbone
        # weights, data order and dropout are identical to a PhaseFormer run with the same seed
        self.cyclespec = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(getattr(configs, "fix_seed", 2021)) + 7919)
            if self.cs_port in ("input", "both", "io"):
                self.cyclespec["input"] = CycleSpecAdapter(
                    feat_dim=self.period_len, axis_len=self.num_periods_input,
                    hidden=int(getattr(configs, "cs_hidden", 8)), windows=windows, periodic=False, **common)
            if self.cs_port in ("block", "both"):
                self.cyclespec["block"] = nn.ModuleList([
                    CycleSpecAdapter(feat_dim=self.latent_dim, axis_len=self.period_len, hidden=block_hidden,
                                     windows=None, periodic=True, **common)
                    for _ in self.routing_layers])
            if self.cs_port in ("output", "io"):
                self.cyclespec["output"] = CycleSpecAdapter(
                    feat_dim=self.period_len, axis_len=self.num_periods_input + self.num_periods_output,
                    hidden=int(getattr(configs, "cs_hidden", 8)), windows=None, periodic=False, **common)

        self._hooks = []
        self._past = None
        if "output" in self.cyclespec:
            # keep the original (pre-adapter) past cycles for the output port; registered first so it
            # runs before the input-port hook
            def stash(mod, args):
                self._past = args[0]
            self._hooks.append(self.embedding.register_forward_pre_hook(stash))
        if "input" in self.cyclespec:
            # PhaseEmbedding input: (B, C, L=24 phases, P_in=30 cycles) -> (B*C, 30, 24)
            def to_ntf(x):
                B, C, L, P = x.shape
                return x.permute(0, 1, 3, 2).reshape(B * C, P, L)

            def from_ntf(y, x):
                B, C, L, P = x.shape
                return y.reshape(B, C, P, L).permute(0, 1, 3, 2)

            self._hooks.append(attach_input_port(self.embedding, self.cyclespec["input"], to_ntf, from_ntf, self._probe))
        if "block" in self.cyclespec:
            # output of norm1 inside each CrossPhaseRoutingLayer: (B*C, 24 phases, D), already (N, T, F)
            for unit, ad in zip(self.routing_layers, self.cyclespec["block"]):
                self._hooks.append(attach_output_port(unit.interact.norm1, ad, lambda x: x, lambda y, x: y, self._probe))

        if "output" in self.cyclespec:
            # predictor output: (B, C, L=24 phases, P_out future cycles); past: (B, C, 24, P_in)
            def out_hook(mod, args, y):
                past = self._past
                B, C, L, Po = y.shape
                seq = torch.cat([past, y], dim=-1)                       # (B, C, 24, P_in + P_out)
                ntf = seq.permute(0, 1, 3, 2).reshape(B * C, seq.shape[-1], L)
                ref = _apply(self.cyclespec["output"], ntf, self._probe)
                ref = ref.reshape(B, C, seq.shape[-1], L).permute(0, 1, 3, 2)
                return ref[..., -Po:]
            self._hooks.append(self.predictor.register_forward_hook(out_hook))

    # ---------------- frozen-backbone protocol ----------------
    def backbone_named_parameters(self):
        for n, p in self.named_parameters():
            if not n.startswith("cyclespec."):
                yield n, p

    def set_backbone_frozen(self, frozen: bool = True):
        self.freeze_backbone = frozen
        for _, p in self.backbone_named_parameters():
            p.requires_grad_(not frozen)
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for name, child in self.named_children():
                if name != "cyclespec":
                    child.eval()
        return self

    # ---------------- diagnostics ----------------
    @torch.no_grad()
    def cyclespec_delta_ratio(self, x_enc):
        """Mean ||delta|| / ||x|| over all attached adapters for one batch."""
        self._probe.active, self._probe.values = True, []
        try:
            self(x_enc)
        finally:
            self._probe.active = False
        v = self._probe.values
        return sum(v) / len(v) if v else 0.0
