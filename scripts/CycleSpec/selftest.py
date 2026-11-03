"""CycleSpec self-test. Run from the repo root:  python scripts/CycleSpec/selftest.py

Checks, on CPU (and on GPU if available):
  1. adapter parameter count (480 for hidden=8, P=24, 3 windows)
  2. every STFT scale reconstructs its input when the conv is skipped (STFT->iSTFT identity)
  3. a freshly built CycleSpec model equals PhaseFormer: same backbone weights, same output,
     same global RNG state afterwards (so data order / dropout match in a joint run)
  4. frozen mode: only adapter params receive gradients and change after an optimizer step
  5. after one step the adapter is no longer the identity (gradient reaches FC_up)
Exits non-zero on the first failure.
"""
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.CycleSpec import CycleSpecAdapter, Model as CycleSpec  # noqa: E402
from models.PhaseFormer import Model as PhaseFormer  # noqa: E402


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


def etth1_cfg(**kw):
    cfg = dict(
        task_name="long_term_forecast", seq_len=720, pred_len=96, enc_in=7, period_len=24,
        phase_layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32,
        phase_num_routers=8, phase_attn_heads=1, phase_attn_dropout=0.1, phase_attn_use_relpos=1,
        phase_attn_window=None, phase_attention_dim=None, phase_use_pos_embed=1, phase_pos_dropout=0.0,
        use_revin=1, revin_affine=0, revin_eps=1e-5, fix_seed=2021,
        cs_hidden=8, cs_windows="30,15,7", cs_kernel=3, cs_branches="both", cs_act="gelu",
    )
    cfg.update(kw)
    return Namespace(**cfg)


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        sys.exit(1)


def run(device):
    print(f"\n=== device: {device} ===")

    # 1. parameter count
    ad = CycleSpecAdapter(period_len=24, num_cycles=30, hidden=8, windows=(30, 15, 7)).to(device)
    n = count_parameters(ad)
    ok(n == 480, f"adapter parameters = {n} (expected 480)")

    # 2. STFT -> iSTFT identity per scale
    u = torch.randn(64, 30, 4, device=device)
    for s in ad.freq_scales:
        rec = s(u, apply_conv=False)
        err = (rec - u).abs().max().item()
        spec = s._stft(u.permute(0, 2, 1).reshape(-1, 30))
        ok(err < 1e-4, f"win={s.win:2d} hop={s.hop} bins={spec.shape[-2]:2d} frames={spec.shape[-1]:2d} recon max err={err:.2e}")

    # 2b. chunking gives the same result (forward and gradients)
    torch.manual_seed(0)
    ad_c = CycleSpecAdapter(period_len=24, num_cycles=30, hidden=8, windows=(30, 15, 7), chunk=7).to(device)
    torch.nn.init.normal_(ad_c.fc_up.weight, std=0.1)
    ad_f = CycleSpecAdapter(period_len=24, num_cycles=30, hidden=8, windows=(30, 15, 7), chunk=0).to(device)
    ad_f.load_state_dict(ad_c.state_dict())
    xc = torch.randn(50, 30, 24, device=device, requires_grad=True)
    xf = xc.detach().clone().requires_grad_(True)
    yc, yf = ad_c(xc), ad_f(xf)
    yc.square().sum().backward()
    yf.square().sum().backward()
    d_out = (yc - yf).abs().max().item()
    d_grad = max((a.grad - b.grad).abs().max().item() for a, b in zip(ad_c.parameters(), ad_f.parameters()))
    ok(d_out < 1e-5 and d_grad < 1e-4, f"chunked == unchunked (output diff {d_out:.1e}, grad diff {d_grad:.1e})")

    # 2c. ECL-sized batch on GPU: 256 x 321 series through forward + backward
    if device.type == "cuda":
        big = CycleSpecAdapter(period_len=24, num_cycles=30, hidden=8, windows=(30, 15, 7)).to(device)
        torch.nn.init.normal_(big.fc_up.weight, std=0.1)
        xb = torch.randn(256 * 321, 30, 24, device=device)
        big(xb).square().mean().backward()
        torch.cuda.synchronize()
        ok(True, f"forward+backward with {256 * 321:,} series (ECL, batch 256, chunk {big.chunk})")
        del big, xb
        torch.cuda.empty_cache()

    # 3. plug == base at init
    cuda = torch.cuda.is_available()
    torch.manual_seed(2021)
    base = PhaseFormer(etth1_cfg()).float()
    rng_after_base = torch.get_rng_state()
    cuda_after_base = torch.cuda.get_rng_state_all() if cuda else []
    torch.manual_seed(2021)
    plug = CycleSpec(etth1_cfg()).float()
    rng_after_plug = torch.get_rng_state()
    cuda_after_plug = torch.cuda.get_rng_state_all() if cuda else []
    ok(torch.equal(rng_after_base, rng_after_plug), "CPU RNG state identical after building base vs plug")
    if cuda:
        ok(all(torch.equal(a, b) for a, b in zip(cuda_after_base, cuda_after_plug)),
           "CUDA RNG state identical after building base vs plug")
    sb, sp = base.state_dict(), plug.state_dict()
    extra = [k for k in sp if k not in sb]
    ok(all(k.startswith("cyclespec.") for k in extra), f"plug adds only adapter keys ({len(extra)} tensors)")
    ok(all(torch.equal(sb[k], sp[k]) for k in sb), "backbone weights identical")
    n_base = sum(p.numel() for p in base.parameters())
    n_plug = sum(p.numel() for p in plug.parameters())
    print(f"        base params = {n_base}, plug params = {n_plug} (+{n_plug - n_base}, +{(n_plug - n_base) / n_base * 100:.1f}%)")

    base, plug = base.to(device).eval(), plug.to(device).eval()
    x = torch.randn(8, 720, 7, device=device) * 3 + 1
    with torch.no_grad():
        yb, yp = base(x), plug(x)
    ok(yp.shape == (8, 96, 7), f"output shape {tuple(yp.shape)}")
    d = (yb - yp).abs().max().item()
    ok(d < 1e-5, f"plug output == base output at init (max diff {d:.2e})")

    # 4. frozen mode
    plug.load_state_dict(base.state_dict(), strict=False)
    plug.set_backbone_frozen(True)
    plug.train()
    ok(not any(m.training for n_, m in plug.named_modules() if n_ and not n_.startswith("cyclespec")),
       "frozen backbone stays in eval mode under model.train()")
    trainable = [n_ for n_, p in plug.named_parameters() if p.requires_grad]
    ok(len(trainable) > 0 and all(n_.startswith("cyclespec.") for n_ in trainable),
       f"only adapter params trainable ({sum(p.numel() for p in plug.parameters() if p.requires_grad)})")
    before = {k: v.clone() for k, v in plug.state_dict().items()}
    opt = torch.optim.Adam([p for p in plug.parameters() if p.requires_grad], lr=1e-3)
    y = torch.randn(8, 96, 7, device=device)
    loss = torch.nn.functional.mse_loss(plug(x), y)
    loss.backward()
    opt.step()
    after = plug.state_dict()
    changed = [k for k in after if not torch.equal(before[k], after[k])]
    ok(len(changed) > 0 and all(k.startswith("cyclespec.") for k in changed),
       f"one Adam step changes only adapter tensors ({len(changed)} changed)")

    # 5. adapter left the identity and sits on the forward path
    ok(plug.cyclespec.fc_up.weight.abs().sum().item() > 0, "FC_up received gradient and moved off zero")
    plug.eval()
    with torch.no_grad():
        y1 = plug(x)
        plug.cyclespec.fc_up.bias.add_(0.5)
        y2 = plug(x)
        plug.cyclespec.fc_up.bias.sub_(0.5)
    ok((y1 - y2).abs().max().item() > 1e-4, "changing the adapter changes the forecast (adapter is on the forward path)")
    r = plug.cyclespec_delta_ratio(x)
    ok(r > 0, f"||Delta||/||X|| after one step = {r:.2e}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    run(torch.device("cpu"))
    if torch.cuda.is_available():
        run(torch.device("cuda"))
    print("\nAll CycleSpec self-tests passed.")
