"""GOOSELocNet on CPU with stub encoders (no weights are downloaded)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

import torch.nn as nn  # noqa: E402

import gooseloc.models.gooselocnet.model as model_mod  # noqa: E402
from gooseloc.models.gooselocnet import (  # noqa: E402
    GOOSELocNet,
    GOOSELocNetConfig,
    heading_from_logits,
    load_checkpoint,
    patch_to_xy,
    predict_batch,
    save_checkpoint,
    total_loss,
)


class StubViT(nn.Module):
    """Returns (B, 5 + N, 1024) tokens from a fixed random projection of pooled pixels."""

    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1024)

    def forward_features(self, x):
        B, _, H, W = x.shape
        patches = nn.functional.avg_pool2d(x, 16).flatten(2).transpose(1, 2)   # (B, N, 3)
        return torch.cat([torch.zeros(B, 5, 1024), self.lin(patches)], dim=1)


class StubUtonia(nn.Module):
    pass


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(model_mod, "build_vit", lambda *a, **k: StubViT())
    monkeypatch.setattr(model_mod, "build_utonia", lambda *a, **k: StubUtonia())
    monkeypatch.setattr(model_mod, "build_utonia_transform", lambda *a, **k: None)

    def fake_lidar(self, points):
        B, K = len(points), self.config.lidar_sectors
        return torch.randn(B, 1224), torch.randn(B, K, 1224)

    monkeypatch.setattr(GOOSELocNet, "encode_lidar", fake_lidar)


def _batch(B=2):
    return {"aerial": torch.rand(B, 3, 1024, 1024), "cameras": torch.rand(B, 4, 3, 64, 160),
            "lidar": [torch.randn(100, 4) for _ in range(B)],
            "gt_xy": torch.tensor([[300.0, 600.0], [700.0, 410.0]])[:B],
            "gt_heading_deg": torch.tensor([10.0, 350.0])[:B],
            "tile_rotation_deg": torch.tensor([0.0, 45.0])[:B], "scene_id": ["a", "b"][:B]}


@pytest.mark.parametrize("variant", ["rgb", "lidar", "rgbl"])
def test_forward_backward(stubs, variant):
    model = GOOSELocNet(GOOSELocNetConfig(variant=variant, lora_rank=0)).train()
    b = _batch()
    out = model(b["aerial"], b["cameras"], b["lidar"])
    assert out["scores"].shape == (2, 64, 64) and out["heading_logits"].shape == (2, 72)
    assert ("fusion_gates" in out) == (variant == "rgbl")
    loss = total_loss(out, b)["loss"]
    loss.backward()
    assert model.proj_sat.weight.grad is not None
    assert all(not p.requires_grad for p in model.vit.parameters())


def test_checkpoint_round_trip(stubs, tmp_path):
    model = GOOSELocNet(GOOSELocNetConfig(variant="rgbl", lora_rank=0)).eval()
    save_checkpoint(tmp_path / "m.pt", model, setting="imu_prior", seed=0)
    loaded, ck = load_checkpoint(tmp_path / "m.pt")
    assert ck["setting"] == "imu_prior"
    for k, v in model.trainable_state_dict().items():
        assert torch.equal(v, dict(loaded.named_parameters())[k].detach())


def test_predict_batch_is_north_up(stubs):
    model = GOOSELocNet(GOOSELocNetConfig(variant="rgb")).eval()
    p = predict_batch(model, _batch(), amp=False)
    assert set(p) == {"scene_id", "x", "y", "heading_deg"}
    assert np.all((p["heading_deg"] >= 0) & (p["heading_deg"] < 360))


def test_decoding():
    logits = torch.full((1, 72), -10.0)
    logits[0, 18] = 10.0                                   # bin 18 = 90 degrees
    assert float(heading_from_logits(logits)) == pytest.approx(90.0, abs=1e-3)
    assert patch_to_xy(torch.tensor([[2.0, 3.0]])).tolist() == [[48.0, 32.0]]
