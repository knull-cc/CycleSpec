"""CycleSpec self-test. Run from the repo root:  python scripts/CycleSpec/selftest.py

Core module (CPU and GPU):
  - STFT -> iSTFT reconstructs its input for every window, reflect and circular padding
  - chunked == unchunked (outputs and gradients)
  - ECL-sized batch (82,176 series) runs forward + backward on GPU
For every PhaseFormer port (input, block, both) and arch (spectral, dense):
  - a fresh CycleSpec model == PhaseFormer: same backbone weights, same output, same RNG state
  - only adapter tensors are trainable / change in frozen mode, frozen backbone stays in eval mode
  - after one step the adapters are no longer the identity and sit on the forward path
Exits non-zero on the first failure.
"""
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.CycleSpec import CycleSpecAdapter, Model as CycleSpec  # noqa: E402
from models.PhaseFormer import Model as PhaseFormer  # noqa: E402


def etth1_cfg(**kw):
    cfg = dict(
        task_name="long_term_forecast", seq_len=720, pred_len=96, enc_in=7, period_len=24,
        phase_layers=3, latent_dim=4, phase_encoder_hidden=16, predictor_hidden=32,
        phase_num_routers=8, phase_attn_heads=1, phase_attn_dropout=0.1, phase_attn_use_relpos=1,
        phase_attn_window=None, phase_attention_dim=None, phase_use_pos_embed=1, phase_pos_dropout=0.0,
        use_revin=1, revin_affine=0, revin_eps=1e-5, fix_seed=2021,
        cs_hidden=8, cs_windows="30,15,7", cs_kernel=3, cs_branches="both", cs_act="gelu",
        cs_port="input", cs_arch="spectral", cs_block_hidden=0, cs_chunk=8192,
    )
    cfg.update(kw)
    return Namespace(**cfg)


def ok(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        sys.exit(1)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def core_tests(device):
    ad = CycleSpecAdapter(feat_dim=24, axis_len=30, hidden=8).to(device)
    ok(nparams(ad) == 480, f"input-port adapter parameters = {nparams(ad)} (expected 480), windows {ad.windows}")
    for periodic, T in ((False, 30), (True, 24)):
        a = CycleSpecAdapter(feat_dim=8, axis_len=T, hidden=4, periodic=periodic).to(device)
        u = torch.randn(64, T, 2, device=device)
        for s in a.freq_scales:
            err = (s(u, apply_conv=False) - u).abs().max().item()
            ok(err < 1e-4, f"T={T} {'circular' if periodic else 'reflect '} win={s.win:2d} hop={s.hop} STFT->iSTFT err={err:.1e}")

    torch.manual_seed(0)
    ad_c = CycleSpecAdapter(feat_dim=24, axis_len=30, hidden=8, chunk=7).to(device)
    torch.nn.init.normal_(ad_c.fc_up.weight, std=0.1)
    ad_f = CycleSpecAdapter(feat_dim=24, axis_len=30, hidden=8, chunk=0).to(device)
    ad_f.load_state_dict(ad_c.state_dict())
    xc = torch.randn(50, 30, 24, device=device)
    yc, yf = ad_c(xc), ad_f(xc)
    yc.square().mean().backward()
    yf.square().mean().backward()
    d_out = (yc - yf).abs().max().item()
    d_grad = max(((a.grad - b.grad).abs().max() / (b.grad.abs().max() + 1e-12)).item()
                 for a, b in zip(ad_c.parameters(), ad_f.parameters()))
    ok(d_out < 1e-5 and d_grad < 1e-4, f"chunked == unchunked (output diff {d_out:.1e}, relative grad diff {d_grad:.1e})")

    if device.type == "cuda":
        big = CycleSpecAdapter(feat_dim=24, axis_len=30, hidden=8).to(device)
        torch.nn.init.normal_(big.fc_up.weight, std=0.1)
        big(torch.randn(256 * 321, 30, 24, device=device)).square().mean().backward()
        torch.cuda.synchronize()
        ok(True, f"forward+backward with {256 * 321:,} series (ECL, batch 256, chunk {big.chunk})")
        del big
        torch.cuda.empty_cache()


def port_tests(device, port, arch):
    print(f"  -- port={port} arch={arch}")
    cuda = device.type == "cuda"
    torch.manual_seed(2021)
    base = PhaseFormer(etth1_cfg()).float()
    rng_b = torch.get_rng_state()
    cuda_b = torch.cuda.get_rng_state_all() if cuda else []
    torch.manual_seed(2021)
    plug = CycleSpec(etth1_cfg(cs_port=port, cs_arch=arch)).float()
    rng_p = torch.get_rng_state()
    cuda_p = torch.cuda.get_rng_state_all() if cuda else []
    ok(torch.equal(rng_b, rng_p) and all(torch.equal(a, b) for a, b in zip(cuda_b, cuda_p)),
       "RNG state identical after building PhaseFormer vs CycleSpec")
    sb, sp = base.state_dict(), plug.state_dict()
    ok(all(k.startswith("cyclespec.") for k in sp if k not in sb) and all(torch.equal(sb[k], sp[k]) for k in sb),
       f"backbone weights identical; adapter params = {nparams(plug.cyclespec)} "
       f"(+{nparams(plug.cyclespec) / nparams(base) * 100:.1f}% of {nparams(base)})")

    base, plug = base.to(device).eval(), plug.to(device).eval()
    x = torch.randn(8, 720, 7, device=device) * 3 + 1
    with torch.no_grad():
        yb, yp = base(x), plug(x)
    ok(yp.shape == (8, 96, 7) and (yb - yp).abs().max().item() < 1e-5,
       f"CycleSpec output == PhaseFormer output at init (max diff {(yb - yp).abs().max().item():.1e})")

    plug.load_state_dict(base.state_dict(), strict=False)
    plug.set_backbone_frozen(True)
    plug.train()
    ok(not any(m.training for n, m in plug.named_modules() if n and not n.startswith("cyclespec")),
       "frozen backbone stays in eval mode under model.train()")
    before = {k: v.clone() for k, v in plug.state_dict().items()}
    opt = torch.optim.Adam([p for p in plug.parameters() if p.requires_grad], lr=1e-2)
    torch.nn.functional.mse_loss(plug(x), torch.randn(8, 96, 7, device=device)).backward()
    opt.step()
    changed = [k for k, v in plug.state_dict().items() if not torch.equal(before[k], v)]
    ok(len(changed) > 0 and all(k.startswith("cyclespec.") for k in changed),
       f"one step changes only adapter tensors ({len(changed)} changed)")
    ups = [m.fc_up for m in plug.cyclespec.modules() if isinstance(m, CycleSpecAdapter)]
    ok(all(u.weight.abs().sum().item() > 0 for u in ups), f"every adapter's FC_up received gradient ({len(ups)} adapters)")
    plug.eval()
    with torch.no_grad():
        y1 = plug(x)
        for u in ups:
            u.bias.add_(0.5)
        y2 = plug(x)
        for u in ups:
            u.bias.sub_(0.5)
    ok((y1 - y2).abs().max().item() > 1e-4, "adapters are on the forward path")
    r = plug.cyclespec_delta_ratio(x)
    ok(r > 0, f"||Delta||/||x|| diagnostic = {r:.2e}")


def run(device):
    print(f"\n=== device: {device} ===")
    core_tests(device)
    for port, arch in (("input", "spectral"), ("block", "spectral"), ("both", "spectral"), ("block", "dense"),
                       ("output", "spectral"), ("io", "spectral")):
        port_tests(device, port, arch)


if __name__ == "__main__":
    print("torch", torch.__version__)
    run(torch.device("cpu"))
    if torch.cuda.is_available():
        run(torch.device("cuda"))
    print("\nAll CycleSpec self-tests passed.")
