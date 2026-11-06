from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gooseloc.constants import DEFAULT_ROOT
from gooseloc.data import io


@pytest.mark.parametrize("ext", [".jpg", ".png"])
def test_image_paths_resolve_either_extension(tmp_path, ext):
    (tmp_path / "aerial").mkdir()
    (tmp_path / "cameras" / "front").mkdir(parents=True)
    img = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    img.save(tmp_path / "aerial" / f"123{ext}")
    img.save(tmp_path / "cameras" / "front" / f"123{ext}")
    assert io.aerial_path(tmp_path, "123") == tmp_path / "aerial" / f"123{ext}"
    assert io.camera_path(tmp_path, "123", "front") == tmp_path / "cameras" / "front" / f"123{ext}"
    assert io.read_image(io.aerial_path(tmp_path, "123")).size == (8, 8)


def test_jpg_preferred_and_missing_defaults_to_jpg(tmp_path):
    (tmp_path / "aerial").mkdir()
    for ext in (".png", ".jpg"):
        (tmp_path / "aerial" / f"1{ext}").touch()
    assert io.aerial_path(tmp_path, "1").suffix == ".jpg"
    assert io.aerial_path(tmp_path, "2") == tmp_path / "aerial" / "2.jpg"


def test_resolve_root_order(tmp_path, monkeypatch):
    monkeypatch.delenv(io.ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(io, "_REPO_DIR", tmp_path / "no_repo")
    with pytest.raises(ValueError, match=DEFAULT_ROOT):
        io.resolve_root(None)
    (tmp_path / DEFAULT_ROOT).mkdir(parents=True)
    assert io.resolve_root(None) == Path.cwd() / DEFAULT_ROOT
    monkeypatch.setenv(io.ROOT_ENV, "/elsewhere")
    assert io.resolve_root(None) == Path("/elsewhere")
    assert io.resolve_root("~/x") == Path("~/x").expanduser()
