"""Downloads siftsmall + the 100K SIFT subset and generates synthetic metadata for both."""
from pathlib import Path
from vecdb.io.dataset import load
from vecdb.io.metadata_gen import generate_uncorrelated, generate_correlated, save_metadata

DATA_DIR = Path("data")


def main() -> None:
    small = load("siftsmall", cache_dir=DATA_DIR)
    print(f"siftsmall: base={small.base.shape} queries={small.queries.shape} gt={small.groundtruth.shape}")

    full = load("sift1m_100k", cache_dir=DATA_DIR)
    print(f"sift1m_100k: base={full.base.shape} queries={full.queries.shape} gt={full.groundtruth.shape}")

    for name, bundle in [("siftsmall", small), ("sift1m_100k", full)]:
        uncorr = generate_uncorrelated(bundle.base.shape[0], seed=0)
        save_metadata(DATA_DIR / f"{name}_meta_uncorrelated.npz", uncorr)
        corr = generate_correlated(bundle.base, n_clusters=100, agree_prob=0.85, seed=0)
        save_metadata(DATA_DIR / f"{name}_meta_correlated.npz", corr)
        print(f"{name}: metadata generated (uncorrelated + correlated)")


if __name__ == "__main__":
    main()
