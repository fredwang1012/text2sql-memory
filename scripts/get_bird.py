"""Download the BIRD dev set into data/bird/ and unpack dev.json, dev_tables.json and dev_databases/.

Usage: py -3.12 -m uv run python scripts/get_bird.py
"""
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bird import db_path, load_questions, safe_join, split, split_path  # noqa: E402
from app.config import settings  # noqa: E402

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
SHA256 = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
SIZE = 346207293


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = f" ({done * 100 // total}%)" if total else ""
            print(f"\rdownloaded {done >> 20} MB{pct}", end="", flush=True)
    print()
    tmp.replace(dest)


def extract_member(zf: zipfile.ZipFile, suffix: str, dest: Path) -> None:
    """Extract the member whose path ends with suffix to dest, ignoring any top-level folder such as dev_20240627/."""
    names = [n for n in zf.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
    if not names:
        raise FileNotFoundError(f"{suffix} not found in archive")
    tmp = dest.with_name(dest.name + ".part")  # rename at the end so an interrupted run leaves no truncated file
    with zf.open(min(names, key=len)) as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp.replace(dest)


def unpack_databases(zf: zipfile.ZipFile, bird: Path) -> None:
    for name in zf.namelist():
        if "__MACOSX" in name or name.endswith("/"):
            continue
        parts = name.replace("\\", "/").split("/")
        if "dev_databases" not in parts:
            continue
        rel = "/".join(parts[parts.index("dev_databases") + 1:])
        target = safe_join(bird / "dev_databases", rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")  # rename at the end so an interrupted run leaves no truncated file
        with zf.open(name) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        tmp.replace(target)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def zip_is_valid(path: Path) -> bool:
    return path.stat().st_size == SIZE and sha256_of(path) == SHA256 and zipfile.is_zipfile(path)


def ensure_zip(path: Path) -> None:
    """Make sure a verified dev.zip is present: a wrong or corrupt file is deleted and fetched once more."""
    if path.exists() and zip_is_valid(path):
        return
    if path.exists():
        print(f"{path} failed verification, deleting")
        path.unlink()
    download(URL, path)
    if not zip_is_valid(path):
        path.unlink()
        raise SystemExit(f"{URL} failed size/SHA-256 verification twice")


def main() -> None:
    bird = settings.bird_dir
    bird.mkdir(parents=True, exist_ok=True)
    zip_path = bird / "dev.zip"
    ensure_zip(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for fname in ("dev.json", "dev_tables.json"):
            if not (bird / fname).exists():
                extract_member(zf, fname, bird / fname)
        if not db_path(settings.db_id).exists():
            inner = bird / "dev_databases.zip"
            if any(n.endswith("dev_databases.zip") for n in zf.namelist()):
                extract_member(zf, "dev_databases.zip", inner)
                with zipfile.ZipFile(inner) as dbz:
                    unpack_databases(dbz, bird)
                inner.unlink()
            else:
                unpack_databases(zf, bird)

    if not db_path(settings.db_id).exists():
        raise SystemExit(f"expected {db_path(settings.db_id)} after extraction")
    questions = load_questions(settings.db_id)
    seed_qs, eval_qs = split(questions, record_to=split_path(settings.db_id))
    print(f"{settings.db_id}: {len(questions)} dev questions ({len(seed_qs)} seed / {len(eval_qs)} eval)")


if __name__ == "__main__":
    main()
