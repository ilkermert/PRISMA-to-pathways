# PRISMA-to-pathways analysis of the large-scale green hydrogen literature

Selected code and settings that accompany a manuscript under review on a PRISMA-based bibliometric study of
large-scale green hydrogen applications. The study combines PRISMA identification, bibliometric analysis,
BERTopic topic modelling, a technology readiness assessment and research and technology development pathways.

This repository holds one file, this README. It describes the analysis and shows the key parts of the code,
so that a reader can see how each step was done. It is not the full codebase and it does not allow the
tables and figures of the paper to be reproduced.

## What is shared and what is not

| Item | Status |
|---|---|
| Search query, search date, filters and record counts | Shared below |
| Model parameters and software versions | Shared below |
| Key code excerpts | Shared below |
| Summary results of the robustness and search sensitivity analyses | In the Supplementary Material of the paper |
| Web of Science records (titles, abstracts, identifiers) | **Not shared.** The records are governed by the database licence and cannot be redistributed |
| Complete analysis code, document-level topic assignments, input workbook | Not public. They form part of ongoing research |

The paper does not claim that its results can be reproduced from the material in this repository.

## Data source

Records were retrieved from the Web of Science Core Collection (all indexes) on 27 July 2026.

```
(hydrogen) [Title] AND
("green hydrogen" OR "renewable hydrogen" OR "water electrolysis" OR "power-to-hydrogen" OR "PEM electrolyzer"
 OR "AEM electrolyzer" OR "SOEC" OR "alkaline electrolysis" OR "electrolytic hydrogen") [Topic] AND
("large-scale" OR "large scale" OR "deployment" OR "scale-up" OR "commercialization" OR "infrastructure"
 OR "roadmap" OR "pathway" OR "outlook" OR "future" OR "economy" OR "supply chain") [Title]
```

The search returned 977 records. It was not restricted by year, language or document type.

| Step | Records |
|---|---|
| Identified | 977 |
| Duplicates removed (UT and DOI) | 0 |
| After document-type filter (Article, Review, including Early Access) | 889 |
| After year filter (2015 to 2026) | 855 |
| After scope-based exclusion (67 excluded) | 788 |
| Relevance score of 7 or higher (final corpus) | 659 |
| Score 4 to 6 (borderline) and score 0 to 2 (weak match), excluded without a recorded manual decision | 122 and 7 |

The relevance score is a heuristic. It counts the co-occurrence of core electrolysis terms, green or renewable
hydrogen terms, scale and deployment terms and future-oriented terms in the title and abstract. It is not a
manual PRISMA screening. The weights and term lists were not archived. A threshold sensitivity analysis is
reported in the paper instead.

## Structure of the analysis

One internal script reads a single Excel workbook and produces the figures, tables and checks of the paper.

| Stage | Content | Paper item | Needs sentence embeddings |
|---|---|---|---|
| `core` | PRISMA counts, bibliometrics, keyword network, topic table, temporal windows, Wilson intervals, tests of topic shares, readiness matrix, pathways figure, evaluator agreement | Tables 1 to 4, Figures 1 to 9, S8, S9, S11, S12, S14, S15 | No |
| `rerun` | Re-estimation of the manuscript BERTopic configuration | Section 3.4 | Yes |
| `robustness` | Parameter, seed, outlier-handling and subsample grid | S1 to S3 | Yes |
| `cohesion` | Cluster cohesion with random baselines | S13 | Yes |
| `threshold` | Relevance-threshold sensitivity | S10, S10b, S10c | Yes |
| `search` | Relaxed-search sensitivity | S4 to S7 | Yes |

Every printed number of the paper that the script can recompute is compared with the value in the manuscript
by a `check()` call. A run ends with a list of passed and failed checks.

### Input workbook (not distributed)

| Sheet | Content |
|---|---|
| `corpus` | The 659 records with the stored BERTopic assignment (`bertopic_topic`) |
| `wos_raw_977` | The original export of 977 records (selected fields) |
| `screening_788` | Relevance-scored list of the 788 eligible records |
| `scope_excluded_67` | Records excluded by the scope rules, with the reason |
| `wos_q4`, `wos_q3a` | Two relaxed searches used for the search sensitivity analysis |
| `table3_authors`, `table4_benchmark` | Tables 3 and 4 of the paper |
| `trl_raters`, `rubric` | Ratings of the two independent external evaluators and the rubric they received |
| `embeddings` (optional) | Sentence embeddings. If absent, they are computed with sentence-transformers |

## Key code

The excerpts below are cut directly from the script.

### Relevance-score bands

`scr` is the scored list of the 788 eligible records and `S_min` is 7. The bands are counted from the
scores. The check lines of the script compare them with the numbers of the paper.

```python
sc = pd.to_numeric(scr["Relevance_Score"])
N_incl = int((sc >= S_min).sum())
N_bord = int(sc.between(4, 6).sum())
N_weak = int((sc <= 2).sum())
```

### BERTopic configuration

The embedding model is `all-MiniLM-L6-v2`. Embeddings are computed from the title and abstract, passed to
BERTopic, and the clustering uses the parameters below. The parameters are the ones reported in the paper.

```python
def make_topic_model(m_cs_=m_cs, k_nn_=k_nn, seed_=s_0):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    return BERTopic(embedding_model=None,
                    umap_model=UMAP(n_neighbors=k_nn_, n_components=d_u, min_dist=0.0, metric="cosine", random_state=seed_),
                    hdbscan_model=HDBSCAN(min_cluster_size=m_cs_, metric="euclidean", cluster_selection_method="eom", prediction_data=True),
                    vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=n_df),
                    calculate_probabilities=False, verbose=False)
```

### Outlier reduction

HDBSCAN leaves a large share of documents unassigned (37.9% in the stored run). When the outlier rate exceeds
0.25, the unassigned documents are reassigned with c-TF-IDF. The paper reports how much this choice
changes the result. Without reassignment, Topic 0 has at least as many documents as Topic 1 in the 2025 window
in every valid re-estimation of the grid (6 to 14 initial topics), whereas Topic 1 is the largest topic in the stored assignment.

```python
tm = make_topic_model()
t0, _ = tm.fit_transform(CORPUS_DOCS, embeddings=emb)
t0 = np.array(t0)
out_rate = float((t0 == -1).mean())
n_init = int(len(set(t0)) - (1 if -1 in t0 else 0))
final = np.array(tm.reduce_outliers(CORPUS_DOCS, t0.tolist(), strategy="c-tf-idf")) if out_rate > r_o else t0.copy()
```

### Intervals and tests of the topic shares

Topic shares in a window are reported with Wilson score intervals. Differences between periods are tested with
Fisher exact tests per topic, a Holm correction across the ten topics, and a permutation test of the whole
topic distribution. The paper treats the overlap of intervals as descriptive and not as a test.

```python
def wilson(k, n, z=z_a):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))
```

```python
def holm(pvals):
    """Holm step-down adjustment."""
    p = np.asarray(pvals, float)
    order = p.argsort()
    adj, prev = np.zeros(len(p)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = prev
    return adj
```

```python
ps = np.array([fisher_exact([[int(v1[t]), n1 - int(v1[t])], [int(v2[t]), n2 - int(v2[t])]])[1] for t in T])
hp = holm(ps)
# permutation test of the chi-square statistic of the whole topic distribution
labels = np.array([t for t in T for _ in range(int(v1[t]))] + [t for t in T for _ in range(int(v2[t]))])
gv = np.array([0] * n1 + [1] * n2)
obs = chi2_contingency(np.array([v1.values, v2.values]) + 1e-9, correction=False)[0]
cnt = 0
for _ in range(N_PERM):
    g_ = rng.permutation(gv)
    tab = np.array([[np.sum((labels == t) & (g_ == k)) for t in T] for k in (0, 1)])
    cnt += (chi2_contingency(tab + 1e-9, correction=False)[0] if (tab.sum(axis=0) > 0).all() else 0) >= obs
pperm = (cnt + 1) / (N_PERM + 1)
```

The same tests are run for five pairs of periods of equal length (Supplementary Table S14). Four of them contain
only complete years and form the main sensitivity analysis. The fifth is the pair of windows of Section 3.5.

### Readiness classification and horizons

Clusters are placed on horizons from their readiness classification only. Publication growth is reported next to
the classification and is not an input to this rule.

```python
def horizon_rule(row):
    """Placement rule of Section 2.6. Growth is NOT an input."""
    typ, trl = str(row["Type"]), _trl_range(str(row["TRL"]).split("(")[0])
    if typ.startswith("Technology") and "context" not in typ and trl:
        if trl[0] >= 7:
            return "2025-2030"
        if trl[1] <= 4:
            return "2030-2035"
        return "no horizon (TRL 5 or 6)"
    if typ in ("Systems", "Policy"):
        return "Enabling condition"
    if typ == "Conceptual":
        return "Guiding framework"
    return "Research indicator"
```

### Cluster cohesion

Cohesion is the mean pairwise cosine similarity between the document embeddings of the members of a cluster.
It is compared with random subsets of the same size.

```python
def pair_stats(idx):
    if len(idx) < 3:
        return len(idx), np.nan, np.nan, np.nan
    v = (emb[idx] @ emb[idx].T)[np.triu_indices(len(idx), k=1)]
    return len(idx), float(v.mean()), float(np.median(v)), float(v.std(ddof=1))
```

### Relevance-threshold sensitivity

Records below the threshold are assigned to the nearest topic centroid of the stored assignment. The threshold
`tau` is the 5th percentile of the similarity between each stored document and its own centroid. It is not a
relevance criterion. This is a centroid-based approximation and not the original topic model.

```python
C = np.vstack([E_fin[topics == t].mean(axis=0) for t in T])
C = C / np.linalg.norm(C, axis=1, keepdims=True)
self_sim = np.array([E_fin[i] @ C[T.index(topics[i])] for i in range(len(fin_))])
tau = float(np.percentile(self_sim, 5))
S_ = E_c @ C.T
cand["assigned_topic"] = [T[j] for j in S_.argmax(axis=1)]
cand["similarity"] = S_.max(axis=1)
cand["inside_structure"] = cand.similarity >= tau
```

## Parameters

| Symbol | Meaning | Value |
|---|---|---|
| `S_min` | Relevance-score threshold | 7 |
| `k_nn` | UMAP n_neighbors | 15 |
| `d_u` | UMAP n_components | 5 |
| `m_cs` | HDBSCAN min_cluster_size | 12 |
| `n_df` | CountVectorizer min_df | 3 |
| `r_o` | Outlier-rate trigger for outlier reduction | 0.25 |
| `s_0` | Random seed | 42 |
| `w_h` | Window half-width in years | 1 |
| `f_min`, `e_min` | Keyword frequency and edge weight thresholds of the keyword network | 25, 2 |
| `z_a` | z-value of the Wilson interval | 1.96 |

Temporal windows are centred on 2015, 2020 and 2025 with a half-width of one year. Growth is compared between
the complete periods 2020 to 2022 and 2023 to 2025.

## Software

Versions used for the BERTopic runs reported in the paper:

```
bertopic==0.17.4
umap-learn==0.5.12
hdbscan==0.8.44
sentence-transformers==5.7.0
scikit-learn==1.6.1
numpy, pandas, scipy, networkx, matplotlib, openpyxl
```

## How to read the results

- The topic partition can differ between software environments, even with a fixed seed. Two re-estimations of
  the manuscript configuration produced different partitions in this study, and the source of the difference
  could not be determined from the available records. The paper therefore reports stability across seeds,
  parameters, outlier handling and subsamples instead of a single run.
- The stored assignment used in the paper cannot be reproduced exactly from this repository.
- Several figures of the paper are typeset versions. A figure drawn by the script from the same data can differ
  in styling.

## Citation

Please cite the paper once it is published. The reference will be added here.

## Licence

[To be confirmed by the authors before release.]
