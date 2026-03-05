import torch
from eb_jepa.losses import VideoJEPA_BCS, VideoJEPA_BCS_Euler_Scaleinvariant


def make_data(dist: str, B=2, C=8, T=2, H=4, W=4, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    if dist == "normal":
        return torch.randn(B, C, T, H, W, generator=g)
    if dist == "uniform":
        return torch.rand(B, C, T, H, W, generator=g) * 2 - 1
    if dist == "shifted":
        return torch.randn(B, C, T, H, W, generator=g) + 2.0
    raise ValueError(dist)


def run_once():
    torch.manual_seed(0)
    variants = [
        ("VideoJEPA_BCS_complex", VideoJEPA_BCS(num_slices=64, lmbd=10)),
        ("VideoJEPA_BCS_euler", VideoJEPA_BCS_Euler_Scaleinvariant(num_slices=64, lmbd=10)),
    ]

    dists = ["normal", "uniform", "shifted"]

    results = {}
    for name, mod in variants:
        mod.eval()
        results[name] = {}
        for dist in dists:
            x = make_data(dist)
            with torch.no_grad():
                out = mod(x)
            # module returns either (loss, bcs_loss, dict) or similar
            if isinstance(out, tuple) and len(out) >= 2:
                loss_val = out[0].item() if hasattr(out[0], "item") else float(out[0])
                bcs_val = out[1].item() if hasattr(out[1], "item") else float(out[1])
            elif isinstance(out, dict) and "bcs_loss" in out:
                loss_val = out.get("loss", float("nan"))
                bcs_val = out["bcs_loss"]
            else:
                loss_val = float("nan")
                bcs_val = float("nan")

            results[name][dist] = (loss_val, bcs_val)

    # Print summary
    print("BCS variants summary (loss, bcs):")
    for name in results:
        print(f"\n{name}:")
        for dist in dists:
            l, b = results[name][dist]
            print(f"  {dist}: loss={l:.6f}, bcs={b:.6f}")

    # Basic sanity checks
    for name in results:
        for dist in dists:
            l, b = results[name][dist]
            assert torch.isfinite(torch.tensor(l)), f"non-finite loss {name} {dist}"
            assert torch.isfinite(torch.tensor(b)), f"non-finite bcs {name} {dist}"


if __name__ == "__main__":
    run_once()
