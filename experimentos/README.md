# Experiments

The scripts that produce each result in the article. They read from `../dados/`, write to
`../resultados/` and — when the folder sits inside the article's monorepo — write the LaTeX
tables and the figures straight into `../../latex/`. Outside it, those tables and figures
go to `../saidas/`. That decision belongs to [`caminhos.py`](caminhos.py); no script builds
a path on its own.

Script and folder names are kept in Portuguese, so that the paths cited in the article, in
the result files and in the code all match.

## Environment

Two environments, because of binary incompatibilities:

```bash
# main — models, baselines, case study
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# figures and tables — need matplotlib, which conflicts with the numpy pin above
pip install matplotlib numpy
```

`estatisticas_corpus.py` runs on the standard library alone, with no installation at all.

`neuralmind/bert-base-portuguese-cased` (BERTimbau) is downloaded from Hugging Face on
first execution. The SentiLex-PT lexicon **does not ship with this repository** for
licensing reasons: download it and place it at `_recursos/SentiLex-flex-PT02.txt` (see
[`REPRODUCAO.md`](REPRODUCAO.md)).

## Scripts

| Script | What it does | Outputs | Cost |
|---|---|---|---|
| `estatisticas_corpus.py` | recomputes size, distribution, Fleiss' $\kappa$, agreement and unanimity of the corpus | `estatisticas_corpus.json` | seconds |
| `baselines.py` | the three baselines of increasing complexity: lexicon, TF-IDF + logistic regression, and frozen BERTimbau as a feature extractor | `baselines_results.json` | minutes |
| `reproduzir_baselines_lexicos.py` | re-runs the lexicon baselines over the full corpus, as a check | `baseline_lexico.json`, `baseline_sentilexpt.json` | seconds |
| `estudo_caso_ordenacao.py` | trains the BERTimbau ensemble (5 folds) and applies it to 40 videos from 4 real searches, comparing the produced order against the order returned by the platform | `estudo_caso_ordenacao.json`, `predicoes_teste.npz` | **~1h50 on CPU** |
| `significancia_e_erros.py` | McNemar and paired bootstrap against the baselines, on the same split; extracts the model's errors | `significancia.json`, `erros_taxonomia.json` and the tables `tab-significancia.tex`, `tab-teste.tex`, `tab-baselines.tex` | ~2 min |
| `erros_categorias.py` | manual categorization of the 72 Positive/Neutral errors, recorded item by item for auditing | `erros_categorias.json` | seconds |
| `gerar_matriz_confusao.py` | confusion matrix figure | `figuras/fig02-confusion-matrix.png` | seconds |
| `avaliar_reranking.py` | evaluates the re-ranking with information retrieval metrics over 29 queries derived from titles, with out-of-fold prediction | `reranking_results.json`, `reranking_curvas.png`, `reranking_tabela.tex` | seconds |
| `ablacao_score.py` | ablation of the scoring function: compares the proposed formulation against three alternatives over the **same** predictions | `ablacao_score.json`, `tab-ablacao.tex` | seconds |
| `gerar_tabelas_estudo_caso.py` | case study tables and figures | `tab-estudo-caso.tex`, `tab-ordem.tex`, two figures | seconds |
| `versao_en.py` | regenerates the tables and figures with English labels, from the same result files | `*-en.tex`, `figuras/en/` | seconds |

**Suggested order**: `estudo_caso_ordenacao.py` first — it is the only expensive one and it
produces `predicoes_teste.npz`, which `significancia_e_erros.py`,
`gerar_matriz_confusao.py` and `erros_categorias.py` consume. Then the rest, in any order.

The result files in `../resultados/` **are versioned**, so that the article's numbers can be
audited without running anything. The `.tex` tables and the figures are not: they are
reproducible at any time and, inside the monorepo, belong to the manuscript.

## Where to check each claim in the article

| Claim | Where to check it |
|---|---|
| 2,749 sentences, 251 videos, distribution 868/999/882 | `estatisticas_corpus.py` → `estatisticas_corpus.json` |
| Fleiss' $\kappa$ = 0.72; 63% unanimous votes | idem |
| Result tables on the held-out test set and per class | `significancia_e_erros.py`, from `predicoes_teste.npz` |
| Confusion matrix | `gerar_matriz_confusao.py`, from the same `predicoes_teste.npz` |
| McNemar and paired bootstrap against the baselines | `significancia_e_erros.py` → `significancia.json` |
| Taxonomy of the 103 errors; the 72 between Positive and Neutral | `significancia_e_erros.py` → `erros_taxonomia.json`; `erros_categorias.py` → `erros_categorias.json` |
| nDCG@k, P@k and MAP of the re-ranking, 29 queries | `avaliar_reranking.py` → `reranking_results.json` |
| Ablation of the scoring function | `ablacao_score.py` → `ablacao_score.json` |
| Case study: scores, AUC between strata, change of order | `estudo_caso_ordenacao.py` → `estudo_caso_ordenacao.json` |
| Baseline reproduction over the full corpus | [`REPRODUCAO.md`](REPRODUCAO.md) |

The three result tables, the confusion matrix and the error analysis all come out of
`predicoes_teste.npz` — a single prediction file, so that divergence between them is
impossible by construction.

## Relationship with the platform

These scripts **do not import** the code in `../plataforma/`: they reimplement the training
and the scoring function in a self-contained way, so that reproduction does not depend on
bringing the application up nor on having a YouTube Data API key. The scoring function is
the same on both sides — Equation (2) of the article — and `ablacao_score.py` is the script
that treats it as an object of study.
