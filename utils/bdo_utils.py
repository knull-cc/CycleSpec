"""Pure-python helpers for the BDO (Boosted Direct Output) head. No torch import on purpose,
so they can be unit-tested anywhere."""
import math


def bdo_stage_periods(p_out, n_stages):
    """Number of output periods covered by each stage (cumulative prefix, last stage = p_out).

    p_out=4, n=3 -> [2, 3, 4];  p_out=30, n=3 -> [10, 20, 30];  n=1 -> [p_out]
    """
    assert p_out >= 1 and n_stages >= 1
    return [min(p_out, max(1, math.ceil(p_out * (j + 1) / n_stages))) for j in range(n_stages)]


def bdo_stage_weights(mode, n_stages):
    """Loss weight of each stage (sums to 1).

    inv   : proportional to 1/(j+1), shorter prefixes weigh more (ReNF's choice, normalised)
    equal : 1/n each
    last  : only the final (full-horizon) stage is supervised -> architecture-only ablation
    """
    if mode == 'inv':
        raw = [1.0 / (j + 1) for j in range(n_stages)]
    elif mode == 'equal':
        raw = [1.0] * n_stages
    elif mode == 'last':
        raw = [0.0] * (n_stages - 1) + [1.0]
    else:
        raise ValueError("bdo_w_mode must be one of inv|equal|last, got %r" % (mode,))
    s = sum(raw)
    return [r / s for r in raw]
