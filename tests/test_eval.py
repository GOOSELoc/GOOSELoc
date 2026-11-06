import numpy as np
import pandas as pd
import pytest

from gooseloc.constants import GSD_M
from gooseloc.data.metadata import select
from gooseloc.eval import evaluate, format_summary, per_scene_errors, summarize
from gooseloc.eval.cli import main as cli_main


def _perfect(scenes, split="test"):
    gt = select(scenes, split)
    return pd.DataFrame({"scene_id": gt.scene_id, "x": gt.gt_px_x.astype(float),
                         "y": gt.gt_px_y.astype(float), "heading_deg": gt.heading_deg})


def test_perfect_predictions(scenes):
    res = evaluate(_perfect(scenes), "test")
    s = res["summary"]
    assert s["all"]["n"] == 1862 and s["all"]["loc_mean_m"] == 0.0 and s["all"]["ori_mean_deg"] < 1e-9
    assert {"paved", "unpaved", "winter", "spring", "summer", "autumn", "test_buffered"} <= set(s)
    assert s["test_buffered"]["n"] == 840
    assert "loc mean" in format_summary(s)


def test_known_errors(scenes):
    pred = _perfect(scenes, "dev")
    pred["x"] += 30.0                                   # 30 px = 6 m east
    pred["y"] -= 40.0                                   # 40 px = 8 m north
    pred["heading_deg"] = (pred["heading_deg"] + 190.0) % 360.0
    err = per_scene_errors(pred, "dev", scenes=scenes)
    np.testing.assert_allclose(err.loc_err_m, 50.0 * GSD_M)
    np.testing.assert_allclose(err.ori_err_deg, 170.0, atol=1e-6)


def test_position_only_and_partial(scenes):
    pred = _perfect(scenes, "dev").drop(columns="heading_deg").iloc[:10]
    with pytest.raises(ValueError):
        per_scene_errors(pred, "dev", scenes=scenes)
    err = per_scene_errors(pred, "dev", scenes=scenes, allow_partial=True)
    assert len(err) == 10 and err.ori_err_deg.isna().all()
    assert "ori_mean_deg" not in summarize(err, scenes=scenes)["all"]


def test_duplicates_rejected(scenes):
    pred = _perfect(scenes, "dev")
    with pytest.raises(ValueError):
        per_scene_errors(pd.concat([pred, pred.iloc[:1]]), "dev", scenes=scenes)


def test_cli(tmp_path, scenes, capsys):
    p = tmp_path / "pred.csv"
    _perfect(scenes, "dev").to_csv(p, index=False)
    cli_main([str(p), "--split", "dev", "--json", str(tmp_path / "s.json")])
    assert "all" in capsys.readouterr().out and (tmp_path / "s.json").exists()
