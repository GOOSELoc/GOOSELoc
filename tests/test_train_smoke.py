"""scripts/train.py end to end on CPU: stub encoders, the synthetic mini dataset, one epoch."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("timm")

from test_model import stubs  # noqa: E402,F401  (fixture)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", ["rgb", "rgbl"])
def test_train_one_epoch(stubs, mini_root, tmp_path, monkeypatch, variant):  # noqa: F811
    cfg = yaml.safe_load((REPO / "configs" / "gooselocnet" / f"{variant}_imu_prior.yaml").read_text())
    cfg.update(epochs=1, eval_every=1, batch_size_per_gpu=1, effective_batch_size=2, num_workers=0, amp=False)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    import gooseloc.models.gooselocnet.inference as inference
    monkeypatch.setattr(inference, "CAMERA_SIZE", (32, 80))           # keep the CPU test fast
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)    # CPU even where a GPU is visible
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(cfg_path), "--root", str(mini_root),
                                      "--out", str(out), "--seed", "0"])
    train.main()
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["dev_metrics"]["n"] == 3 and ck["setting"] == "imu_prior"
