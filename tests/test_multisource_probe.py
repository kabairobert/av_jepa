import torch
import torch.nn as nn

from eb_jepa.jepa import JEPAProbe, MultiSourceJEPAProbe


class _DummyJEPA(nn.Module):
    def __init__(self):
        super().__init__()

    def get_features(self, observations):
        return {
            "encoder": observations,
            "projector": observations + 1.0,
        }

    def route_state(self, state, source):
        if source == "encoder":
            return state
        if source == "projector":
            return state + 1.0
        raise ValueError(source)


class _DummyDetHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    def score(self, preds, targets):
        out = []
        for pred in preds:
            p = self.forward(pred).mean().abs()
            t = targets.mean().abs()
            out.append(float((p / (t + 1e-6)).item()))
        return out


class _DummyPixHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_multisource_forward_with_source_losses_matches_forward_mean():
    torch.manual_seed(0)
    jepa = _DummyJEPA()
    obs = torch.randn(2, 3, 4, 5, 5)

    enc_probe = JEPAProbe(jepa, _DummyPixHead(3, 3), nn.MSELoss(), feature_source="encoder")
    proj_probe = JEPAProbe(jepa, _DummyPixHead(3, 3), nn.MSELoss(), feature_source="projector")
    multi = MultiSourceJEPAProbe(
        {"encoder": enc_probe, "projector": proj_probe},
        active_source="encoder",
    )

    mean_loss, by_source = multi.forward_with_source_losses(obs, obs)
    forward_loss = multi(obs, obs)

    assert set(by_source.keys()) == {"encoder", "projector"}
    expected = (by_source["encoder"] + by_source["projector"]) / 2
    assert torch.allclose(mean_loss, expected)
    assert torch.allclose(forward_loss, expected)


def test_multisource_score_by_source_returns_both_sources():
    torch.manual_seed(0)
    jepa = _DummyJEPA()
    obs = torch.randn(2, 3, 4, 5, 5)
    targets = torch.randn(2, 1, 2, 5, 5)
    preds = [torch.randn(2, 3, 2, 5, 5), torch.randn(2, 3, 2, 5, 5)]

    enc_probe = JEPAProbe(jepa, _DummyDetHead(3), nn.BCEWithLogitsLoss(), feature_source="encoder")
    proj_probe = JEPAProbe(jepa, _DummyDetHead(3), nn.BCEWithLogitsLoss(), feature_source="projector")
    multi = MultiSourceJEPAProbe(
        {"encoder": enc_probe, "projector": proj_probe},
        active_source="projector",
    )

    out = multi.score_by_source(preds, targets, pred_source="encoder")

    assert set(out.keys()) == {"encoder", "projector"}
    assert len(out["encoder"]) == len(preds)
    assert len(out["projector"]) == len(preds)
