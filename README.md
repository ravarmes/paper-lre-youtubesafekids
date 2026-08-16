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
| Equation (2) — displayed suitability reading $A$ | `plataforma/app/filters/sentiment.py`, `_adequacy` |
| Equation (3) — Safety Score from predicted class, confidence and $A$ | `plataforma/app/filters/sentiment.py`, `process` method |
| Equation (4) — aggregation of $N$ filters by the mean | `plataforma/app/filters/__init__.py`, `FilterManager.process_video` |
| Equation (5) — suitability of a video from its human labels (evaluation only) | `experimentos/avaliar_reranking.py` |
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

2,749 sentences from 264 children's videos in Brazilian Portuguese, annotated into three
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

Distribution: 872 Negative (31.7%), 988 Neutral (35.9%), 889 Positive (32.3%).
Agreement: Fleiss' $\kappa$ = 0.7180 (*substantial agreement*), 81.2% mean pairwise
agreement and 62.5% unanimous votes (1,718 of 2,749). Running
`experimentos/estatisticas_corpus.py` recomputes all of this straight from the file.

The four annotators cover all 2,749 sentences, with no gaps. Exactly 2 sentences (0.1%)
split the panel 2×2 and were settled by the arbiter; on every other item the final label is
the simple majority.

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

Held-out test set, 545 sentences (20%, stratified), over the 2,722 sentences used for
training and evaluation:

| Method | Accuracy | Macro F1 | F1 Negative |
|---|---:|---:|---:|
| SentiLex-PT | 46.24 | 45.86 | 46.9 |
| TF-IDF + logistic regression | 66.06 | 66.57 | 79.8 |
| Frozen BERTimbau + LR | 72.11 | 72.26 | 82.5 |
| **Fine-tuned BERTimbau (*ensemble*)** | **78.17** | **78.53** | **88.0** |

The 88.0% F1 on the Negative class (84.9% recall) is the critical metric: in a child
protection use case, letting negative content through is a qualitatively more serious error
than deprioritizing suitable content. The decisive comparison is against frozen BERTimbau,
which shares model, tokenization and representation with the proposed system: the 6.26
points of macro F1 between them isolate what domain adaptation adds, with a 95% confidence
interval of [2.91, 9.73] under a paired bootstrap.

Execution details, a suggested order and the full map of which script backs which claim are
in [`experimentos/README.md`](experimentos/README.md).

## Safety Score

The input is a single block concatenating title, description and three transcript excerpts
(beginning, middle and end), which captures the narrative arc without submitting the whole
transcript to the model. The ensemble returns a distribution over the three classes, and two
quantities are read from it. The first is the **predicted suitability** $A$:

$$A = P(\text{Positive}) + 0.5 \cdot P(\text{Neutral})$$

The second is the **Safety Score** $S$, which is what orders the listing. Writing $C$ for the
confidence of the predicted class:

$$S = \begin{cases} 0.10 + 0.20\,(1-C) & \text{if the class is Negative} \\ 0.70 + 0.15\,A & \text{otherwise} \end{cases}$$

The principle is that **risk dominates tone**. The image is two disjoint bands — $[0.10,
0.233]$ for Negative and $[0.70, 0.85]$ for the rest — and the gap between them, $0.467$, is
more than three times the $0.15$ over which the upper band varies. Ordering by tone therefore
operates strictly *within* content that presents no risk, and no distribution of
probabilities can lift a Negative video above a non-Negative one. That is a property of the
algebra, not a calibrated tolerance.

Inside the safe band $A$ grades: between two videos that present no risk, the one with
positive affective charge is placed ahead of the merely informative one. What the formulation
refuses is not the gradation but its promotion to a risk signal — putting Neutral *between*
the bands would turn the absence of positive charge into evidence of risk.

Three readings, three roles: the **position** in the listing comes from Equation (3) and
answers risk; the **color** of the indicator comes from the predicted class and says what
the item is; the **percentage** displayed is Equation (2), the same $A$ that grades the safe
band — so among non-demoted items the percentages read top to bottom in descending order.

`experimentos/ablacao_score.py` measures what this formulation is worth against the
alternatives it rejects, over the same out-of-fold predictions and the same 32 queries.
Flattening the safe band to a constant costs 0.037 of nDCG@3 (0.719 against 0.755) and leaves
80 distinct score values where the full formulation produces 263. Placing Neutral as
intermediate risk costs 0.042 of nDCG@3 and 0.035 of MAP (0.713 and 0.847, against 0.755 and
0.882). Ordering by the expectation $A$ alone reproduces every metric to the third decimal:
what the bands buy is a worst-case guarantee, not an average, which is why their justification
comes from the case study and not from these ranking metrics.

## Model

The classifier is **BERTimbau Base (cased)** fine-tuned on the corpus, aggregated into an
*ensemble* by soft voting over the five models of the stratified cross-validation.
Hyperparameters: 5 epochs, batch size 8, learning rate $3\times10^{-5}$, 100 warm-up steps,
Random Oversampling applied only to the training split of each fold.

**The weights are not versioned** — neither for the experiments nor for the platform.
`experimentos/estudo_caso_ordenacao.py` retrains from scratch; on CPU this takes about 22
minutes per fold.

## License

Three components, three terms — see [`LICENSE`](LICENSE) for the binding text.

| Component | Terms |
|---|---|
| Code (`plataforma/`, `experimentos/`) and result files | MIT |
| Annotation: the four independent labels, the final label, the rubric | CC BY 4.0 |
| Transcribed sentences, video titles and descriptions | third-party material, quoted for research; **no licence granted** |

The split matters and is not a formality: the annotation is the authors' own work and can be
licensed, whereas the transcribed speech belongs to the video owners and is reproduced only
to the extent needed to verify the results. The SentiLex-PT lexicon is a third-party resource
under its own terms and is **not** redistributed here; `experimentos/README.md` records where
to obtain it.

## Citation

See `CITATION.cff`. The manuscript is under review; the full reference and the DOI will be
added after acceptance.
