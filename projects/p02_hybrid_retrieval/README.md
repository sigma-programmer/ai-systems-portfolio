# P02 — Hybrid Reranking Retrieval Stack

BM25 sparse retrieval + dense vector search, fused with RRF or weighted
score fusion, reranked by a learned cross-encoder-style model (or any LLM),
with pre-retrieval metadata filtering and a latency-vs-accuracy sweep that
produces a Pareto frontier. All from scratch on numpy; runs fully offline.

## Problem

Neither retriever alone covers real query traffic:

* **Sparse (BM25)** nails exact identifiers (`0xae7bf7`, `PN48213`) thanks to
  IDF, but has no notion of synonyms or typos (`reimbursement` ≠ `refund`,
  `Veltrra` ≠ `Veltra`).
* **Dense** embeddings generalize across wording and spelling but dilute a
  single rare token among generic words, so pasted error codes get lost.

A production stack fuses both, spends a reranking budget on the top of the
list, and lets you pick the operating point on the latency/accuracy curve.

## Architecture

```
                      metadata filter (DSL / Mongo-style JSON)
                                   │  compiled to a bitmask (cached)
                                   ▼
            ┌───────────────► BM25 inverted index ─────────┐  top candidate_k
  query ────┤                 (precomputed impacts)          │
            │                                               ▼
            └──► HashingEmbedder ─► Flat | IVF(nprobe) ──► fusion ──► top rerank_depth ──► reranker ──► top-k
                                                          RRF | weighted       │
                                                          min-max              ├─ CrossEncoderReranker
                                                                               │   11 query-doc features
                                                                               │   + numpy logistic regression
                                                                               └─ LLMReranker (aikit.LLM, JSON)

  offline:  corpus.py (240 docs, 240 graded queries) ─► evaluation.py (R@k, MRR, nDCG@k)
            ─► tuning.py (train reranker, sweep configs, Pareto frontier) ─► bench.py
```

| Module | What it does |
|---|---|
| `bm25.py` | Okapi BM25 (k1=1.2, b=0.75, smoothed idf). Inverted index whose postings store the precomputed BM25 *impact* per (term, doc), so a query is a few numpy scatter-adds. Deterministic top-k with tie-breaking. |
| `dense.py` | `FlatIndex` (exact, one matvec) and `IVFIndex` (spherical k-means with k-means++ seeding, contiguous per-list storage, `nprobe` knob). |
| `filters.py` | `Eq`, `In`, `Range`, `And`, `Or`, `Not` with `&`/`\|`/`~` operators, plus `parse_filter` for `{"$and": [...], "year": {"$gte": 2021}, "source": {"$in": [...]}}` (also `$ne`, `$nin`, `$not`). `MetadataStore` keeps a per-field value→bitmap index; masks are applied *inside* both retrievers (pre-filtering, so filtered queries still get a full candidate pool). |
| `fusion.py` | Reciprocal Rank Fusion and weighted fusion over min-max / max / z-score normalized scores. Per-source contributions are kept for explainability. |
| `features.py` | Pairwise features: log BM25, BM25 relative to the best candidate, dense cosine (absolute and relative), query-term coverage, IDF-weighted coverage, bigram match, proximity (smallest window that covers the matched terms), fuzzy coverage (best char-trigram Jaccard against the vocabulary, which tolerates typos), title coverage, and title fuzzy match. |
| `logreg.py` | L2-regularized logistic regression via Newton/IRLS, with standardization and balanced class weights. **scikit-learn isn't available in this environment, so this replaces `sklearn.linear_model.LogisticRegression`.** It converges in about 9 Newton steps and is deterministic. |
| `rerank.py` | `CrossEncoderReranker` (learned weights; hand-set defaults when untrained) and `LLMReranker` (one listwise prompt, JSON scores, safe fallback). Neither one ever drops a candidate. |
| `pipeline.py` | `HybridSearcher` + pydantic `SearchConfig` (mode, fusion, alpha, rrf_k, candidate_k, flat/ivf, nprobe, rerank, rerank_depth). Returns hits with per-stage scores and per-stage timings. |
| `corpus.py` | The bundled labeled benchmark (see below). |
| `evaluation.py`, `tuning.py`, `bench.py` | Metrics, reranker training, the sweep and Pareto frontier, IVF scaling, filtering, and LLM prompt-size accounting. |

### The bundled benchmark

The corpus has 6 product domains × 5 fictional products × 8 support aspects (pricing,
refund, login, errors, performance, start-up, deletion, purchase), giving **240
documents**. Each document combines aspect templates, randomized specifics
(prices, day counts, unique hex error codes, part numbers), domain filler, and
sometimes a mention of a sibling product as a hard negative. There are **240
queries** in four families of 60:

| Family | Example | Expected winner |
|---|---|---|
| `keyword` | `seeing 0xae7bf7 again since yesterday` | sparse (rare-token IDF) |
| `paraphrase` | `remove my data on Ostrm` (typo + synonym of "delete") | dense (char n-grams + concepts) |
| `mixed` | `how quick is Sprintly Cloud` (exact product + synonym of "latency") | needs both |
| `lexical` | `where to purchase Taskwell Teams` | everyone |

Relevance is graded. The target document is grade 2. Other documents about the same product
are grade 1 ("right product, wrong question"). Recall and MRR count grade ≥ 2,
and nDCG uses the full grades. Queries are split 50/50 per family into train
(used only to fit the reranker) and test (used for every number below).

## Results (measured with `bench.py`; latencies are from an Apple-silicon laptop and accuracy is deterministic)

MRR on the 120 held-out test queries, broken down by family:

| config | overall | keyword | lexical | mixed | paraphrase | p50 latency |
|---|---|---|---|---|---|---|
| sparse (BM25) | 0.589 | **1.000** | 1.000 | 0.318 | 0.038 | 0.014 ms |
| dense (flat) | 0.604 | 0.160 | 1.000 | **0.983** | 0.271 | 0.072 ms |
| hybrid RRF, c50 | 0.638 | 0.624 | 1.000 | 0.664 | 0.262 | 0.076 ms |
| hybrid weighted α=0.5, c50 | 0.725 | 0.728 | 1.000 | 0.900 | 0.272 | 0.090 ms |
| hybrid weighted + rerank@20 | 0.892 | 1.000 | 1.000 | 0.911 | 0.657 | 0.35 ms |
| **hybrid weighted + rerank@40** | **0.952** | 1.000 | 1.000 | 0.911 | **0.897** | 0.63 ms |

For that last configuration, recall@1 is 0.925, recall@10 is 0.992, and nDCG@10 is 0.803.

**Pareto frontier (nDCG@10 vs p50 latency):**
`sparse/c10` (0.699, 0.014 ms) → `hybrid/w0.5/c10` (0.702, 0.048 ms) →
`hybrid/w0.5/c50/rr10` (0.748, 0.24 ms) → `hybrid/w0.5/c50/rr20` (0.756, 0.35 ms) →
`hybrid/rrf/c50/rr40` (0.804, 0.55 ms). The other 13 configs in the sweep are dominated.

Other measurements:

* **Metadata filtering.** Pre-filtering on the target's domain raises hybrid MRR from 0.725 to 0.842 and
  *lowers* latency (0.089 to 0.073 ms) because fewer documents are scored. Across 600 filtered results there were 0 filter violations.
* **IVF scaling** (50k clustered vectors, 128-d, 128 lists). Exact search takes 0.88 ms per query.
  With `nprobe=1`, recall@10 is 0.937 at 0.039 ms (23× faster, 0.9% of vectors scanned). With `nprobe=4`, it is
  0.961 at 0.054 ms. With `nprobe=32`, it is 0.989 at 0.30 ms. On the 240-document corpus IVF barely affects
  latency, so the knob only pays off at scale. That is why the scaling test exists.
* **Reranker training.** Training on 120 queries × 30 candidates takes about 100 ms and 9 Newton iterations.
  The biggest standardized weights are `title_fuzzy` (+6.0), `idf_coverage` (+4.4),
  `bm25_rel` (+3.7), and `dense_rel` (+2.9). `term_coverage` and `title_coverage` come out negative
  because they are collinear with their IDF-weighted and fuzzy counterparts.
* **LLM rerank cost model.** A depth-20 listwise prompt is about 1,970 tokens on average
  (`aikit.count_tokens`). Multiply by your model's input price to compare it with the
  learned reranker, which costs nothing and takes under a millisecond.
* The full benchmark runs in about 2 s, and the 48 tests run in about 0.7 s.

## Key design decisions and tradeoffs

* **No stemming in BM25, on purpose.** The gap between exact and fuzzy matching is what makes
  hybrid retrieval worth having. Stemming would hide part of that gap behind a language-specific heuristic.
* **RRF vs weighted fusion.** RRF needs no score calibration, but with large
  candidate pools it ranks "appears in both lists" above "rank 1 in one list". Keyword queries
  (BM25 rank 1, absent from dense) drop from MRR 1.0 to 0.62, and mixed queries drop to 0.66.
  Min-max weighted fusion keeps a dominant single-retriever hit near the top. α=0.7 (dense-heavy)
  falls back to 0.27 on keyword queries, so α is a real tuning knob and it is in the sweep.
* **Rerank depth is the main accuracy lever.** The first stage has recall@40 ≈ 0.99, but
  rank 1 is often wrong for typo'd product names. Going from depth 10 to depth 40 raises paraphrase MRR from 0.53
  to 0.90, at a cost of about 2.5× the latency.
* **Pre-filtering, not post-filtering.** Masks go into the BM25 and dense top-k, so a selective
  filter can't empty the candidate pool. Masks are cached by canonical filter JSON.
* **Precomputed BM25 impacts.** These trade index memory (one float per posting) for
  query-time speed, the same idea as impact-ordered indexes.
* **The rerankers can't break retrieval.** They only reorder the head of the list.
  `LLMReranker` falls back to the first-stage order on unparsable output, and passages the LLM
  leaves unscored sink below scored ones but stay in the list.
* **Remaining errors are honest.** The 5 mixed-query misses are cases where the reranker
  prefers the start-up document, whose template repeats the product name twice. That is a realistic
  "keyword stuffing" failure, and the benchmark keeps it.

## How to run

```bash
# tests (48)
.venv/bin/python -m pytest projects/p02_hybrid_retrieval

# demo (offline; uses Claude for the LLM reranker step if ANTHROPIC_API_KEY is set and AIKIT_OFFLINE is not)
AIKIT_OFFLINE=1 .venv/bin/python projects/p02_hybrid_retrieval/demo.py

# benchmark (JSON)
cd projects/p02_hybrid_retrieval && ../../.venv/bin/python -m hybrid_retrieval.bench
```

Library use:

```python
from hybrid_retrieval import HybridSearcher, SearchConfig, Eq, Range, load_benchmark, split_queries, train_reranker

docs, queries = load_benchmark()
s = HybridSearcher(docs)
train_reranker(s, split_queries(queries)[0])
resp = s.search("Veltra X2 reimbursement", k=5,
                config=SearchConfig(fusion="weighted", rerank=True, rerank_depth=40),
                filter=Eq("domain", "automotive") & Range("year", gte=2020))
print(resp.doc_ids, resp.timings_ms)
```
