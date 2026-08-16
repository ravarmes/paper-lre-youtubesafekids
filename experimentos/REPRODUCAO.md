# Baseline reproduction — 2026-07-30

Re-run of the *baselines* over the **full corpus of 2,749 sentences**, as an independent
check on the numbers of the conference version of this work.

> **What this document is and is not.** It audits the baselines *as published in the
> conference version*, over an 80/20 split of the whole 2,749-sentence corpus. **None of the
> numbers on this page appear in the journal manuscript.** The baselines reported there are
> different ones: they run on the 2,722 sentences that remain after the 27 belonging to
> case-study videos are removed, over the same 545-sentence test split as the ensemble, and
> they come out of `significancia_e_erros.py` into `significancia.json`. The two sets are not
> comparable, which is why the ones on this page do not match the README or the article. For
> the journal numbers, see the table in the root [`README.md`](../README.md).

## Environment

The system-wide Python has `numpy 2.0.2`, incompatible with the `pandas` and
`scikit-learn` binaries installed alongside it — both fail on import with
`ValueError: numpy.dtype size changed`. The fix was an isolated virtual environment with
the versions pinned in `requirements.txt`:

```
.venv/              numpy 1.26.4 · pandas 2.1.3 · scikit-learn 1.3.2
                    torch 2.8.0+cpu · transformers 4.37.2
```

To recreate it, if needed:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "numpy==1.26.4" "pandas==2.1.3" "scikit-learn==1.3.2"
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install "transformers==4.37.2"
```

`.venv/` and `_recursos/` are in `.gitignore` — they are regenerable and take up space.

## Inputs

- `corpus.csv` — 2,749 sentences (872 Negative / 988 Neutral / 889 Positive), 264 videos.
- `_recursos/SentiLex-flex-PT02.txt` — SentiLex-PT, a polarity lexicon for Portuguese
  (6.9 MB), **not redistributed** in this repository: download it from the official
  SentiLex-PT distribution and place it at that path. **70,215 forms** (19,031 positive,
  51,184 negative) after consolidation by `POL:N0`.
- The `neuralmind/bert-base-portuguese-cased` model, from the local Hugging Face cache.

Split: 80/20 stratified, `random_state=42` over the whole corpus → **2,199 train / 550
test**. This is the split of the conference version, *not* the one used in the journal
manuscript, which first removes the 27 case-study sentences and then splits 2,722 into
2,177 / 545.

## Commands

```powershell
.venv\Scripts\python.exe reproduzir_baselines_lexicos.py   # simple lexicon + SentiLex-PT
.venv\Scripts\python.exe baselines.py                      # TF-IDF+LR + frozen BERTimbau
```

## Result

| Baseline | Metric | Conference version | Reproduced | Δ |
|---|---|---:|---:|---:|
| **SentiLex-PT** | accuracy | 45.45 | **45.45** | **0.00** |
| | macro F1 | 44.68 | **44.68** | **0.00** |
| | F1 Neg / Neu / Pos | 47.1 / 48.2 / 38.7 | **47.10 / 48.22 / 38.73** | **0.00** |
| **TF-IDF + LR** | accuracy | 69.45 | **69.45** | **0.00** |
| | macro F1 | 69.66 | 69.68 | +0.02 |
| | F1 Neg / Neu / Pos | 80.2 / 65.3 / 63.4 | 80.23 / 65.00 / 63.82 | ±0.4 |
| **Frozen BERTimbau** | accuracy | 74.73 | 74.36 | −0.37 |
| | macro F1 | 74.81 | 74.51 | −0.30 |
| | F1 Neg / Neu / Pos | 88.0 / 69.6 / 66.9 | 88.83 / 67.84 / 66.86 | ±1.8 |

### Reading

**SentiLex-PT reproduces exactly**, in every published digit — including per class. This
confirms both the published numbers and the attribution of the baseline to the SentiLex-PT
of Silva et al. (2012).

**TF-IDF reproduces accuracy exactly**; macro F1 differs by 0.02 percentage points, a
variation without practical meaning, attributable to the `scikit-learn` version.

**Frozen BERTimbau differs by 0.3 to 0.4 percentage points.** This is the expected outcome
for a baseline that depends on embedding extraction: the value varies with the
`torch`/`transformers` version, with the device (this run was on CPU) and with the
preprocessing applied before the embedding. That sensitivity was already documented in the
originating project, in a baseline script not included here which compares three
preprocessing variants and records the range **73.82% – 74.73%**. The value obtained here
(74.36%) falls inside that range.

**Conclusion**: the conference version's baselines are reproducible. Its published numbers
were kept; the differences in the frozen BERTimbau are environmental, not methodological.
The journal manuscript re-runs all three baselines on its own split, and those are the
numbers it reports.

## A note on the "simple lexicon"

`baselines.py` implements, as baseline 1, a lexicon of positive and negative words of its
own (the `lexicon_predict` function), which **is not** SentiLex-PT and **is not a baseline
reported in either version of the article**. Reproduced here, it reaches 43.64% accuracy and 39.48%
macro F1 — below SentiLex-PT, as expected from a much smaller manual list.

It appears in `baselines_results.json` because the script runs it alongside the others; the
article's lexicon baseline is the one in `baseline_sentilexpt.json`.

## Generated files

| File | Content |
|---|---|
| `baseline_sentilexpt.json` | SentiLex-PT — the lexicon baseline, as run here |
| `baseline_lexico.json` | simple lexicon — auxiliary baseline, in neither version of the article |
| `baselines_results.json` | simple lexicon, TF-IDF+LR and frozen BERTimbau |
| `reproduzir_baselines_lexicos.py` | script for the lexicon baselines, adapted from the originating project |
