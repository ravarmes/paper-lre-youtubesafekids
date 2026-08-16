# YouTube Safe Kids — sentiment analysis for children's video curation

Supporting material for the article *"A Sentiment-Annotated Corpus of Brazilian Portuguese Children's Video Transcripts: Annotation Protocol, Classifier and Use in Search Result Re-ranking"*.

It brings together the three things needed to verify the work: the **platform** described
in the article, the annotated **corpus** and the **experiments** that produce every
reported number.

The system classifies the emotional polarity of children's YouTube videos from title,
description and transcript excerpts, converts the prediction into an interpretable
**Safety Score** and uses that score to **re-rank** search results — without removing them
from the listing.

> **Language.** The corpus is in Brazilian Portuguese and stays that way: it is the object
> of study, and the article's claims can only be audited against the original sentences.
> The documentation is in English. `dados/corpus-rubrica.md` is kept in the original
> Portuguese, since the annotation was carried out against that wording, with an English
> translation alongside it in `dados/corpus-rubrica-en.md`.

> **Authorship.** This repository provides the research artifact accompanying the manuscript
> submitted to *Language Resources and Evaluation* (Springer Nature).
> Authors: Rafael Vargas Mesquita dos Santos, João Victor de Salles, Flávio Izo, Sabrina Vargas.
> Repository: [https://github.com/ravarmes/paper-lre-youtubesafekids](https://github.com/ravarmes/paper-lre-youtubesafekids).

## Structure

```
plataforma/     the YouTube Safe Kids application: search, filters, score and re-ranking
experimentos/   the scripts that produce the article's results
dados/          corpus.csv, reordenacao_buscas.csv and the annotation rubric
resultados/     saved outputs (.json, .npz, .png) — allow auditing without running anything
```

Each folder has its own README. The split follows what each part answers: `plataforma/`
shows **how the system works**, `experimentos/` shows **why the article's numbers are what
they are**.

Folder and file names are kept in Portuguese throughout, so that paths cited in the
article, in the result files and in the code all match.

## Where the article touches the code

| In the article | In the repository |
|---|---|
| Equation (1) — single input block (title + description + 3 excerpts) | `plataforma/app/filters/sentiment.py`, `process` method |
| Equation (2) — Safety Score from predicted class and confidence | `plataforma/app/filters/sentiment.py`, same method |
| Equation (3) — displayed suitability reading | `plataforma/app/filters/sentiment.py`, `_adequacy` |
| Equation (4) — aggregation of $N$ filters by the mean | `plataforma/app/filters/__init__.py`, `FilterManager.process_video` |
| Section *Prototype architecture* — client/server, filter as score producer | `plataforma/app/main.py`, `plataforma/app/api/endpoints/videos.py`, `plataforma/app/filters/base.py` |
| Interface figures | `plataforma/app/templates/`, `plataforma/app/static/` |
| Re-ranking without removal from the listing | `plataforma/app/api/endpoints/videos.py`, the ordering by `final_score` |
| Corpus, Fleiss' $\kappa$, class distribution | `experimentos/estatisticas_corpus.py` |
| Result tables, confusion matrix, significance, errors | `experimentos/significancia_e_erros.py`, `experimentos/gerar_matriz_confusao.py`, `experimentos/erros_categorias.py` |
| Re-ranking evaluation and score ablation | `experimentos/avaliar_reranking.py`, `experimentos/ablacao_score.py` |
| Case study on real searches | `experimentos/estudo_caso_ordenacao.py`, `experimentos/gerar_tabelas_estudo_caso.py` |

The same scoring function appears on both sides: the platform applies it at search time,
and `experimentos/ablacao_score.py` compares it against three alternative formulations.

## Data

### `dados/corpus.csv` — annotated corpus

2,749 sentences from 251 children's videos in Brazilian Portuguese, annotated into three
polarity classes.

| Column | Content |
|---|---|
| `ID` | YouTube video identifier |
| `TITULO` | video title |
| `FRASE` | transcribed sentence |
| `PARTE` | position in the narrative: `Início`, `Meio` or `Fim` (beginning, middle, end) |
| `LINK` | video URL |
| `AS` | **final label**, by majority vote: `Negativo`, `Neutro` or `Positivo` |
| `P1`, `A1`, `A2`, `A3` | individual labels of the four annotators (`P1` = faculty member, arbiter in case of a tie) |

Distribution: 868 Negative (31.6%), 999 Neutral (36.3%), 882 Positive (32.1%).
Agreement: Fleiss' $\kappa$ = 0.7218 (*substantial agreement*), 81.5% mean pairwise
agreement and 63.0% unanimous votes (1,731 of 2,749). Running
`experimentos/estatisticas_corpus.py` recomputes all of this straight from the file.

The four annotators cover all 2,749 sentences, with no gaps. No item produced a 2×2 tie —
the final label coincides with the simple majority on all 2,749 sentences, so the
tie-breaking rule through `P1` never had to be invoked.

`dados/corpus-rubrica.md` is the annotation rubric: it defines the three polarity classes,
with an example each, and states the unit of judgment. `dados/corpus-rubrica-en.md` is an
English translation, provided as a reading aid — the annotation itself was carried out
against the Portuguese original.

### `dados/reordenacao_buscas.csv` — case study videos

40 videos returned by 4 real YouTube searches, in two strata: **A** (queries typical of a
caregiver — the order should not be changed) and **B** (adult animation with child appeal —
should be demoted).

| Column | Content |
|---|---|
| `ESTRATO` | `A_responsavel` or `B_apelo_infantil` |
| `TERMO` | query that returned the video |
| `POSICAO` | order in which the platform returned the video |
| `VIDEO_ID` | video identifier |
| `TITULO`, `DESCRICAO` | video metadata |
| `FRASE_INICIO`, `FRASE_MEIO`, `FRASE_FIM` | three transcript excerpts |

Two of these 40 videos also appear in the corpus; the 27 corresponding sentences are
removed from training (2,749 → 2,722) before any split of the data, so that all 40 remain
out of sample.

### Use of the data

The sentences, titles and descriptions are transcribed or copied from public third-party
videos, collected for research purposes. The texts are reproduced only to the extent needed
to verify the results, and the rights over that content remain with their owners (see
`LICENSE`). The corpus contains no personal data of annotators or of platform users.

**There is a single corpus in this repository.** The application keeps no copy of its own:
its `app/nlp/config.py` points to `dados/corpus.csv`.

## Main results

Held-out test set, 544 sentences (20%, stratified), over the 2,722 sentences used for
training and evaluation:

| Method | Accuracy | Macro F1 | F1 Negative |
|---|---:|---:|---:|
| SentiLex-PT | 45.22 | 44.08 | 41.8 |
| TF-IDF + logistic regression | 67.46 | 67.77 | 79.2 |
| Frozen BERTimbau + LR | 75.74 | 76.11 | 87.9 |
| **Fine-tuned BERTimbau (*ensemble*)** | **81.07** | **81.50** | **90.7** |

The 90.7% F1 on the Negative class is the critical metric: in a child protection use case,
letting negative content through is a qualitatively more serious error than deprioritizing
suitable content.

Execution details, a suggested order and the full map of which script backs which claim are
in [`experimentos/README.md`](experimentos/README.md).

## Safety Score

From the predicted class and the model confidence $C$:

$$S = \begin{cases} 0.1 + 0.2\,(1-C) & \text{if the class is Negative} \\ 0.85 & \text{otherwise} \end{cases}$$

The score answers **risk**, not tone: Neutral and Positive receive the same value, because
the distinction between them says something about tone and nothing about risk. Only the
Negative class is demoted, and there confidence grades how far — the more certain the
model, the lower the score.

The input is a single block concatenating title, description and three transcript excerpts
(beginning, middle and end), which captures the narrative arc without submitting the whole
transcript to the model.

Because Equation (2) deliberately stops grading what it does not demote, the interface
displays a separate, graded suitability reading, computed over the ensemble probabilities:

$$A = P(\text{Positive}) + 0.5 \cdot P(\text{Neutral})$$

Three readings, three roles: the **position** in the listing comes from Equation (2) and
answers risk; the **color** of the indicator comes from the predicted class and says what
the item is; the **percentage** displayed is Equation (3) and says how suitable the filter
considers the video.

`experimentos/ablacao_score.py` measures what this formulation is worth against the
alternatives it rejects. Separating Neutral from Positive costs 0.079 of nDCG@3 (0.669
against 0.748) and rewrites the order in 22 of the 29 queries; a continuous score based on
the expectation reaches 0.726, below the piecewise function, because the extra granularity
ends up ordering by tone.

## Model

The classifier is **BERTimbau Base (cased)** fine-tuned on the corpus, aggregated into an
*ensemble* by soft voting over the five models of the stratified cross-validation.
Hyperparameters: 5 epochs, batch size 8, learning rate $3\times10^{-5}$, 100 warm-up steps,
Random Oversampling applied only to the training split of each fold.

**The weights are not versioned** — neither for the experiments nor for the platform.
`experimentos/estudo_caso_ordenacao.py` retrains from scratch; on CPU this takes about 22
minutes per fold.

## License

MIT — see `LICENSE`, including the reservation about the third-party content in the corpus
and about the SentiLex-PT lexicon, which is not redistributed here.

## Citation

See `CITATION.cff`. The manuscript is under review; the full reference and the DOI will be
added after acceptance.
