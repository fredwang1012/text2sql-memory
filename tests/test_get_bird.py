import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "get_bird.py"
spec = importlib.util.spec_from_file_location("get_bird", SCRIPT)
get_bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_bird)


def test_unpack_databases_rejects_traversal_and_extracts_normal_members(tmp_path):
    archive = tmp_path / "dbs.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dev_databases/shop/shop.sqlite", b"data")
        zf.writestr("dev_databases/../evil.txt", b"bad")
    bird = tmp_path / "bird"
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="unsafe"):
            get_bird.unpack_databases(zf, bird)
    # the normal member comes first in the archive, so it was written, whole and without a leftover .part file
    assert (bird / "dev_databases" / "shop" / "shop.sqlite").read_bytes() == b"data"
    assert not list(bird.rglob("*.part"))
    assert not (bird / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
