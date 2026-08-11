from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tarfile
import numpy as np
import requests
from tqdm import tqdm

from vecdb.io.fvecs import read_fvecs, read_ivecs

SIFTSMALL_URL = "http://corpus-texmex.irisa.fr/siftsmall.tar.gz"
SIFT1M_URL = "http://corpus-texmex.irisa.fr/sift.tar.gz"


@dataclass
class DatasetBundle:
    base: np.ndarray        # (N, d) float32
    queries: np.ndarray     # (Q, d) float32
    groundtruth: np.ndarray  # (Q, k) int32, unfiltered top-k neighbour ids


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def _extract(archive: Path, into: Path) -> None:
    if into.exists() and any(into.iterdir()):
        return
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(into)


def download_siftsmall(cache_dir: Path) -> Path:
    archive = cache_dir / "siftsmall.tar.gz"
    _download(SIFTSMALL_URL, archive)
    out = cache_dir / "siftsmall"
    _extract(archive, out)
    return out / "siftsmall"


def download_sift1m(cache_dir: Path) -> Path:
    archive = cache_dir / "sift.tar.gz"
    _download(SIFT1M_URL, archive)
    out = cache_dir / "sift1m"
    _extract(archive, out)
    return out / "sift"


def _subset_100k(base: np.ndarray, groundtruth: np.ndarray, n: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Truncate base to the first n rows; drop groundtruth ids that fall outside the subset
    from each row (do NOT clip/relabel them — a dropped id must not silently become a wrong id)."""
    new_base = base[:n]
    new_gt = [row[row < n] for row in groundtruth]
    max_len = max(len(r) for r in new_gt)
    padded = np.full((groundtruth.shape[0], max_len), -1, dtype=np.int32)
    for i, row in enumerate(new_gt):
        padded[i, : len(row)] = row
    # trim to the shortest common length so the array stays rectangular and valid everywhere
    min_len = min(len(row) for row in new_gt)
    return new_base, padded[:, :min_len]


def load(name: str, cache_dir: Path = Path("data")) -> DatasetBundle:
    cache_dir = Path(cache_dir)
    if name == "siftsmall":
        root = download_siftsmall(cache_dir)
        base = read_fvecs(root / "siftsmall_base.fvecs")
        queries = read_fvecs(root / "siftsmall_query.fvecs")
        gt = read_ivecs(root / "siftsmall_groundtruth.ivecs")
        return DatasetBundle(base=base, queries=queries, groundtruth=gt)
    if name == "sift1m_100k":
        root = download_sift1m(cache_dir)
        base = read_fvecs(root / "sift_base.fvecs")
        queries = read_fvecs(root / "sift_query.fvecs")
        gt = read_ivecs(root / "sift_groundtruth.ivecs")
        base, gt = _subset_100k(base, gt, n=100_000)
        return DatasetBundle(base=base, queries=queries, groundtruth=gt)
    raise ValueError(f"unknown dataset name: {name!r}")
