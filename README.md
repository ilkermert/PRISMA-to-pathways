# Linking Green Hydrogen Research Themes to Technology Readiness: A PRISMA-Based Bibliometric Study and Research Roadmap in Support of 2030–2050 Targets

Authors: İlker Mert, Hüseyin Yağlı, Victor M. Ambriz-Diaz*

This repository contains representative code excerpts and methodological documentation for the research article (under review):

"From Research Trends to Deployment Pathways: A PRISMA-Based Bibliometric Study and Roadmap for Large-Scale Green Hydrogen Applications Toward 2050"

It follows the manuscript's own section order: each subsection below states the methodological component as described in the paper, immediately followed by a representative code excerpt implementing it.

> **Note on scope:** These are representative excerpts of the core methodological steps, included for transparency and to show how each step was done. The complete analysis script, the Web of Science records, the document-level topic assignments and the input workbook are not included, see "Contact and Data Availability" below. This repository does not allow the tables and figures of the paper to be reproduced, and the paper does not claim that it does.

## About the Study

The study combines a PRISMA-based identification of Web of Science records, a bibliometric performance analysis, BERTopic topic modelling, a technology readiness assessment and the construction of research and technology development pathways for large-scale green hydrogen applications. The pathways are derived from the literature. They are not deployment forecasts.

| Stage | Content | Needs sentence embeddings |
|---|---|---|
| `core` | PRISMA counts, bibliometrics, keyword network, topic table, temporal windows, tests of topic shares, readiness matrix, pathways figure, evaluator agreement | No |
| `rerun` | Re-estimation of the manuscript BERTopic configuration | Yes |
| `robustness` | Parameter, seed, outlier-handling and subsample grid | Yes |
| `cohesion` | Cluster cohesion with random baselines | Yes |
| `threshold` | Relevance-threshold sensitivity | Yes |
| `search` | Relaxed-search sensitivity | Yes |

## 2.1 Identification and Screening (PRISMA)

Records were retrieved from the Web of Science Core Collection (all indexes) on 27 July 2026, following PRISMA 2020 guidance [1]. The search was not restricted by year, language or document type.

```
(hydrogen) [Title] AND
("green hydrogen" OR "renewable hydrogen" OR "water electrolysis" OR "power-to-hydrogen" OR "PEM electrolyzer"
 OR "AEM electrolyzer" OR "SOEC" OR "alkaline electrolysis" OR "electrolytic hydrogen") [Topic] AND
("large-scale" OR "large scale" OR "deployment" OR "scale-up" OR "commercialization" OR "infrastructure"
 OR "roadmap" OR "pathway" OR "outlook" OR "future" OR "economy" OR "supply chain") [Title]
```

| Step | Records |
|---|---|
| Identified | 977 |
| Duplicates removed (UT and DOI) | 0 |
| After document-type filter (Article, Review, including Early Access) | 889 |
| After year filter (2015 to 2026) | 855 |
| After scope-based exclusion (67 excluded) | 788 |
| Relevance score of 7 or higher (final corpus) | 659 |
| Score 4 to 6 (borderline) and score 0 to 2 (weak match), excluded without a recorded manual decision | 122 and 7 |

The relevance score is a heuristic. It counts the co-occurrence of core electrolysis terms, green or renewable hydrogen terms, scale and deployment terms and future-oriented terms in the title and abstract. It is not a manual PRISMA screening. The weights and term lists were not archived, so a threshold sensitivity analysis is reported in the paper instead (see "Sensitivity analyses" below). The bands are counted from the scores of the 788 eligible records:

```python
sc = pd.to_numeric(scr["Relevance_Score"])
N_incl = int((sc >= S_min).sum())
N_bord = int(sc.between(4, 6).sum())
N_weak = int((sc <= 2).sum())
```

## 2.2 Bibliometric Performance Analysis

Author and index keywords are merged and counted once per record with pandas [2]. Keywords with at least 25 occurrences become nodes, co-occurrence in a record gives an edge with a minimum weight of 2, and communities are found with the Louvain method [3] as implemented in NetworkX [4].

```python
freq = {}
for ks in doc_kws:
    for k in ks:
        freq[k] = freq.get(k, 0) + 1
keep = {k for k, v in freq.items() if v >= f_min}
ew = {}
for ks in doc_kws:
    for a, b in itertools.combinations(sorted(set(k for k in ks if k in keep)), 2):
        ew[(a, b)] = ew.get((a, b), 0) + 1
G = nx.Graph()
for k in sorted(keep):
    G.add_node(k, freq=freq[k])
for (a, b), w in ew.items():
    if w >= e_min:
        G.add_edge(a, b, weight=w)
G.remove_nodes_from(list(nx.isolates(G)))
comms = sorted(louvain_communities(G, weight="weight", seed=s_0, resolution=1.0), key=len, reverse=True)
```

## 2.3 BERTopic Topic Modelling

Topics are extracted with BERTopic [5]. The embedding model is `all-MiniLM-L6-v2` [6], used within the Sentence-BERT framework [7]. Dimension reduction uses UMAP [8] and clustering uses HDBSCAN [9]. Embeddings are computed from the title and abstract and passed to BERTopic. The clustering parameters are the ones reported in the paper.

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

HDBSCAN leaves a large share of documents unassigned (37.9% in the stored run). When the outlier rate exceeds 0.25, the unassigned documents are reassigned with c-TF-IDF. Without reassignment, Topic 0 has at least as many documents as Topic 1 in the 2025 window in every valid re-estimation of the grid (6 to 14 initial topics), whereas Topic 1 is the largest topic in the stored assignment.

```python
tm = make_topic_model()
t0, _ = tm.fit_transform(CORPUS_DOCS, embeddings=emb)
t0 = np.array(t0)
out_rate = float((t0 == -1).mean())
n_init = int(len(set(t0)) - (1 if -1 in t0 else 0))
final = np.array(tm.reduce_outliers(CORPUS_DOCS, t0.tolist(), strategy="c-tf-idf")) if out_rate > r_o else t0.copy()
```

Cluster cohesion is the mean pairwise cosine similarity between the document embeddings of the members of a cluster. It is compared with random subsets of the same size.

```python
def pair_stats(idx):
    if len(idx) < 3:
        return len(idx), np.nan, np.nan, np.nan
    v = (emb[idx] @ emb[idx].T)[np.triu_indices(len(idx), k=1)]
    return len(idx), float(v.mean()), float(np.median(v)), float(v.std(ddof=1))
```

## 2.4 Temporal Topic Evolution

Topic shares are computed in windows centred on 2015, 2020 and 2025 with a half-width of one year. They are reported with Wilson score intervals. The paper treats the overlap of intervals as descriptive and not as a formal test.

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

Differences between periods are tested with Fisher exact tests per topic, a Holm correction across the ten topics, and a permutation test of the whole topic distribution.

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

The same tests are run for five pairs of periods of equal length. Four of them contain only complete years and form the main sensitivity analysis. The fifth is the pair of windows of Section 3.5. Growth is compared between the complete periods 2020 to 2022 and 2023 to 2025 and is reported for description only.

## 2.5 Technology Readiness Level and Systems/Market Readiness Assessment

Clusters were rated by the authors. Two independent external evaluators then rated all ten clusters with the same written rubric on a blind form. Topic 0 was rated as one cluster, so its sub-clusters 0a and 0b are the authors' classification. The paper reports raw agreement and does not report kappa statistics.

```python
n_type = int((A["Type"] == B["Type"]).sum())
num = lambda s: pd.to_numeric(s, errors="coerce")
both_trl = num(A["TRL"]).notna() & num(B["TRL"]).notna()
n_trl, d_trl = int(((num(A["TRL"]) == num(B["TRL"])) & both_trl).sum()), int(both_trl.sum())
both_rd = num(A["Readiness"]).notna() & num(B["Readiness"]).notna()
n_rd, d_rd = int(((num(A["Readiness"]) == num(B["Readiness"])) & both_rd).sum()), int(both_rd.sum())
```

## 2.6 Pathway Construction

Clusters are placed on horizons from their readiness classification only. Publication growth is reported next to the classification and is not an input to this rule. Clusters on policy, market, infrastructure and supply chain topics are shown as enabling conditions that run in parallel.

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

## Sensitivity Analyses (Sections 3.4 and 3.5)

Records below the relevance threshold are assigned to the nearest topic centroid of the stored assignment. The similarity threshold `tau` is the 5th percentile of the similarity between each stored document and its own centroid. It is not a relevance criterion. This is a centroid-based approximation and not the original topic model.

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

The `robustness` stage varies HDBSCAN `min_cluster_size` (8, 12, 16), UMAP `n_neighbors` (10, 15, 25), five random seeds and three outlier treatments, and repeats the manuscript configuration on 20 subsamples of 80% of the records. The `search` stage assigns records from two relaxed Web of Science queries to the stored topic structure.

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
| `f_min`, `e_min` | Keyword frequency and edge weight thresholds | 25, 2 |
| `z_a` | z-value of the Wilson interval | 1.96 |

## Data Sources

### Web of Science Records

The records (titles, abstracts, identifiers) were exported from the Web of Science Core Collection. They are governed by the database licence and cannot be redistributed, so they are not part of this repository.

### Input Workbook (not distributed)

| Sheet | Content |
|---|---|
| `corpus` | The 659 records with the stored BERTopic assignment |
| `wos_raw_977` | The original export of 977 records (selected fields) |
| `screening_788` | Relevance-scored list of the 788 eligible records |
| `scope_excluded_67` | Records excluded by the scope rules, with the reason |
| `wos_q4`, `wos_q3a` | Two relaxed searches for the search sensitivity analysis |
| `table3_authors`, `table4_benchmark` | Tables 3 and 4 of the paper |
| `trl_raters`, `rubric` | Ratings of the two independent external evaluators and the rubric they received |
| `embeddings` (optional) | Sentence embeddings. If absent, they are computed with sentence-transformers |

## Software Environment

Core stage (local run):

| Package | Version | Role |
|---|---|---|
| Python | 3.12.3 | Core implementation |
| NumPy | 2.4.4 | Array computation and random-number generation |
| SciPy | 1.17.1 | Fisher exact and chi-square tests |
| pandas | 3.0.2 | Record handling |
| NetworkX | 3.6.1 | Keyword network and Louvain communities |
| Matplotlib | 3.10.8 | Figure generation |
| openpyxl | 3.1.5 | Workbook input and output |

BERTopic runs reported in the paper:

| Package | Version | Role |
|---|---|---|
| BERTopic | 0.17.4 | Topic modelling |
| UMAP (umap-learn) | 0.5.12 | Dimension reduction |
| HDBSCAN | 0.8.44 | Clustering |
| sentence-transformers | 5.7.0 | Embeddings (`all-MiniLM-L6-v2`) |
| scikit-learn | 1.6.1 | Vectorizer and agreement metrics |

## How to Read the Results

- The results of topic modelling depend on the software environment, even with a fixed seed. For this reason the paper does not rest on a single run. It reports how the findings behave across seeds, parameter settings, outlier handling and data subsamples (Supplementary Tables S1 to S3), and it describes the topic proportions as properties of the title-restricted corpus.
- Two re-estimations of the manuscript configuration gave different partitions, and the source of the difference could not be determined from the available records. The paper reports this in Sections 3.4 and 5.4.
- The excerpts in this repository illustrate the method.

## References

[1] Page MJ, McKenzie JE, Bossuyt PM, Boutron I, Hoffmann TC, Mulrow CD, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 2021;372:n71.

[2] McKinney W. Data structures for statistical computing in Python. Proc 9th Python Sci Conf 2010:51-6.

[3] Blondel VD, Guillaume J-L, Lambiotte R, Lefebvre E. Fast unfolding of communities in large networks. J Stat Mech 2008;2008:P10008.

[4] Hagberg A, Schult D, Swart P. Exploring network structure, dynamics, and function using NetworkX. Proc 7th Python Sci Conf (SciPy2008) 2008:11-5.

[5] Grootendorst M. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794 2022.

[6] Wang W, Wei F, Dong L, Bao H, Yang N, Zhou M. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Adv Neural Inf Process Syst 2020;33:5776-88.

[7] Reimers N, Gurevych I. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proc 2019 Conf Empir Methods Nat Lang Process (EMNLP) 2019.

[8] McInnes L, Healy J, Melville J. UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426 2018.

[9] McInnes L, Healy J. Accelerated hierarchical density based clustering. 2017 IEEE Int Conf Data Min Workshops (ICDMW) 2017:33-42.

## Citation

If you use this code or methodology in your research, please cite:

```
Mert, İ., Yağlı, H., & Ambriz-Diaz, V. M. (2026). Linking Green Hydrogen Research Themes to Technology Readiness: A PRISMA-Based Bibliometric Study and Research Roadmap in Support of 2030–2050 Targets. International Journal of Hydrogen Energy.
```

## License

This code is made available under the MIT License.

## Contact and Data Availability

The Web of Science records cannot be shared or redistributed because of the database licence. The complete analytical codebase, the document-level topic assignments and the input workbook are not publicly available because they form part of ongoing research. Summary results of the robustness analysis and of the search sensitivity analysis are provided in the Supplementary Material of the paper.


Repository maintained by: İlker Mert (ilkermert@osmaniye.edu.tr)
For questions regarding the methodology, please contact:
İlker Mert, ilkermert@osmaniye.edu.tr

---

This README accompanies representative code excerpts for the article referenced above. Last updated: November 2026
