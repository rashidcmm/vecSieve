# Phase 1 Milestone: Setup

## What got built

Project scaffold with data loaders, metadata generation, and a download script. We built `vecdb.io.fvecs` (binary SIFT format readers for .fvecs and .ivecs), `vecdb.io.dataset` (load function for siftsmall and 100K SIFT subset via FTP), and `vecdb.io.metadata_gen` (synthetic metadata generation in both uncorrelated and correlated flavours). The `scripts/download_data.py` CLI wraps these together as an entry point for the development workflow, downloading both datasets and generating all metadata in one pass.

## Numbers

**siftsmall** (fast iteration dataset):
- base: (10,000, 128)
- queries: (100, 128)
- ground truth: (100, 100)

**sift1m_100k** (headline benchmark):
- base: (100,000, 128)
- queries: (10,000, 128)
- ground truth: (10,000, 0)

**Download and metadata generation timing:**
- 100K SIFT download (~168 MB compressed): ~1 minute 37 seconds
- k-means (n_clusters=100) for correlated metadata: included in above, both datasets metadata generated successfully

## Gate status

✓ PASS. All dimensions match the Day 0 spec exactly. siftsmall.base.shape = (10000, 128), siftsmall.queries.shape = (100, 128), siftsmall.groundtruth.shape = (100, 100), sift1m_100k.base.shape = (100000, 128), sift1m_100k.queries.shape = (10000, 128). The binary format parsers read the correct byte counts and maintain vector dimensionality across all three file types (base, query, ground truth).

## Interview note

We use SIFT from TEXMEX (not custom embeddings) because it is a standard benchmark with published FAISS baselines, making our results directly comparable and reproducible. Synthetic metadata (not true attributes) is generated in two versions — uncorrelated and correlated via k-means clustering — because these stress-test the planner differently. Uncorrelated attributes scatter matching nodes evenly across the graph, letting predicate-aware traversal win. Correlated attributes create spatially contiguous regions; greedy descent can get stranded in zero-match zones, making pre-filtering win instead. Designing both cases upfront means the crossover chart will show where each strategy dominates and *why*, rather than a single clean win on one scenario. This is the highest signal-per-hour investment in the experimental design (§4.2): we get two opposite behaviours, two different insights, and one compelling proof that the planner matters.

