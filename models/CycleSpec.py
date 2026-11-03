"""CycleSpec = PhaseFormer + a Frame2Freq-style spectral adapter on the cycle axis.

Select it with  --model CycleSpec  (the base model stays  --model PhaseFormer).
models/PhaseFormer.py is left untouched; this file subclasses it.

Where the adapter sits
    RevIN -> pad -> split into cycles x_periods (B, C, T=num cycles, P=period_len)
          -> [CycleSpec adapter]  -> Linear(T -> D) phase embedding -> ... (unchanged PhaseFormer)

Adapter, per (sample, variable), on X in R^{T x P}:
    U      = GELU(X W_d + b_d)                       W_d: P -> hidden (24 -> 8)
    U      = [U_f | U_t]                             split hidden in half
    U_f'   = mean_k iSTFT_wk( DWConv_k( STFT_wk(U_f) ) ),   STFT along the cycle axis, w in {30,15,7}
    U_t'   = DWConv1d_3(U_t)                         along the cycle axis
    X'     = X + [U_f' | U_t'] W_u + b_u             W_u, b_u initialised to zero

Because FC_up starts at zero, a freshly built CycleSpec model computes exactly the same
function as PhaseFormer. The adapter is created after the backbone and under a forked
RNG, so with the same seed the backbone weights, the data order and dropout masks are
identical to a PhaseFormer run: the two runs differ only by the adapter.

Choices Frame2Freq leaves open (mine): periodic Hann window, hop = win // 4, centre +
reflect padding; frequency conv of kernel 3 along the frequency-bin axis, depthwise,
real and imaginary parts share weights, one conv per scale; GELU after FC_down.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.PhaseFormer import Model as PhaseFormer


class SpectralScale(nn.Module):
    """One STFT scale: STFT over the cycle axis -> depthwise conv over frequency bins
    (real and imaginary parts share weights) -> iSTFT back to length T."""

    def __init__(self, channels: int, win: int, kernel: int = 3):
        super().__init__()
        self.channels = channels
        self.win = int(win)
        self.hop = max(1, self.win // 4)
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)

    def _stft(self, x):  # (M, T) -> (M, F, frames), complex
        return torch.stft(x, n_fft=self.win, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)

    def _istft(self, spec, length):  # (M, F, frames) -> (M, T)
        return torch.istft(spec, n_fft=self.win, hop_length=self.hop, win_length=self.win,
                           window=self.window, center=True, normalized=False,
                           onesided=True, length=length)

    def forward(self, u, apply_conv: bool = True):  # u: (N, T, c)
        N, T, c = u.shape
        spec = self._stft(u.permute(0, 2, 1).reshape(N * c, T))  # (N*c, Fq, Fr)
        if apply_conv:
            Fq, Fr = spec.shape[-2], spec.shape[-1]
            spec = spec.reshape(N, c, Fq, Fr).permute(0, 3, 1, 2).reshape(N * Fr, c, Fq)
            spec = torch.complex(self.conv(spec.real.contiguous()), self.conv(spec.imag.contiguous()))
            spec = spec.reshape(N, Fr, c, Fq).permute(0, 2, 3, 1).reshape(N * c, Fq, Fr)
        y = self._istft(spec, T)
        return y.reshape(N, c, T).permute(0, 2, 1)


class CycleSpecAdapter(nn.Module):
    """Residual spectral adapter on the cycle axis. Input/output: (N, T, P)."""

    def __init__(self, period_len: int = 24, num_cycles: int = 30, hidden: int = 8,
                 windows: Sequence[int] = (30, 15, 7), kernel: int = 3,
                 branches: str = "both", act: str = "gelu"):
        super().__init__()
        assert branches in ("both", "freq", "time"), branches
        assert act in ("gelu", "none"), act
        self.branches, self.act = branches, act
        if branches == "both":
            self.c_freq = hidden // 2
            self.c_time = hidden - self.c_freq
        elif branches == "freq":
            self.c_freq, self.c_time = hidden, 0
        else:
            self.c_freq, self.c_time = 0, hidden
        self.windows = [min(int(w), num_cycles) for w in windows]

        self.fc_down = nn.Linear(period_len, hidden)
        self.freq_scales = nn.ModuleList(
            [SpectralScale(self.c_freq, w, kernel) for w in self.windows] if self.c_freq > 0 else [])
        self.time_conv = (nn.Conv1d(self.c_time, self.c_time, kernel, padding=kernel // 2, groups=self.c_time)
                          if self.c_time > 0 else None)
        self.fc_up = nn.Linear(hidden, period_len)
        nn.init.zeros_(self.fc_up.weight)
        nn.init.zeros_(self.fc_up.bias)

    def delta(self, x):  # (N, T, P) -> (N, T, P)
        u = self.fc_down(x)
        if self.act == "gelu":
            u = F.gelu(u)
        parts = []
        if self.c_freq > 0:
            u_f = u[..., : self.c_freq]
            parts.append(torch.stack([s(u_f) for s in self.freq_scales], dim=0).mean(dim=0))
        if self.c_time > 0:
            u_t = u[..., self.c_freq:]
            parts.append(self.time_conv(u_t.permute(0, 2, 1)).permute(0, 2, 1))
        return self.fc_up(torch.cat(parts, dim=-1))

    def forward(self, x):
        return x + self.delta(x)


class Model(PhaseFormer):
    """PhaseFormer with the CycleSpec adapter in front of the phase embedding."""

    def __init__(self, configs):
        super().__init__(configs)  # identical backbone, identical RNG consumption
        self.use_cyclespec = True
        self.freeze_backbone = False

        windows = getattr(configs, "cs_windows", "30,15,7")
        if isinstance(windows, str):
            windows = [int(w) for w in windows.split(",") if w.strip()]
        with torch.random.fork_rng(devices=[]):
            # seed the CPU generator only (torch.manual_seed would also reseed CUDA,
            # which fork_rng(devices=[]) does not restore)
            torch.default_generator.manual_seed(int(getattr(configs, "fix_seed", 2021)) + 7919)
            self.cyclespec = CycleSpecAdapter(
                period_len=self.period_len,
                num_cycles=self.num_periods_input,
                hidden=int(getattr(configs, "cs_hidden", 8)),
                windows=windows,
                kernel=int(getattr(configs, "cs_kernel", 3)),
                branches=getattr(configs, "cs_branches", "both"),
                act=getattr(configs, "cs_act", "gelu"),
            )

    # PhaseFormer.forward calls self._to_phase_series(x_periods) exactly once, right after
    # RevIN, padding and the split into cycles, i.e. before Linear(T -> D). Overriding it
    # puts the adapter at that point without copying PhaseFormer.forward.
    def _to_phase_series(self, x_periods):  # (B, C, T, P) -> (B, C, P, T)
        B, C, T, P = x_periods.shape
        x_periods = self.cyclespec(x_periods.reshape(B * C, T, P)).reshape(B, C, T, P)
        return PhaseFormer._to_phase_series(x_periods)

    # ---------------- frozen-backbone protocol ----------------
    def backbone_named_parameters(self):
        for n, p in self.named_parameters():
            if not n.startswith("cyclespec."):
                yield n, p

    def set_backbone_frozen(self, frozen: bool = True):
        """Freeze every PhaseFormer weight; only the adapter stays trainable. The frozen
        backbone is kept in eval mode (no dropout), i.e. it is a fixed function."""
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
    def cycle_view(self, x_enc):
        """RevIN-normalised input as cycles, (B, C, T, P), exactly as PhaseFormer.forward builds it."""
        x_in = self.revin.normalize(x_enc)[0] if self.use_revin else x_enc.float()
        x = x_in.permute(0, 2, 1)
        B, C, _ = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")
        return x.reshape(B, C, self.num_periods_input, self.period_len)

    @torch.no_grad()
    def cyclespec_delta_ratio(self, x_enc):
        """||Delta|| / ||X|| on the cycle view of a batch."""
        x = self.cycle_view(x_enc)
        B, C, T, P = x.shape
        x = x.reshape(B * C, T, P)
        return (self.cyclespec.delta(x).norm() / (x.norm() + 1e-12)).item()
