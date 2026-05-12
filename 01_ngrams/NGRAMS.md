# 🔤 Unit 1 — N-Gram Language Models

**Status:** 🟡 In progress — tutorial script available; exercises pending.

## Learning goal

Build a language model from nothing but counts: estimate the probability of the next token from the frequencies of token sequences seen in a corpus. This is the simplest possible answer to "what is meaning?" — meaning as co-occurrence statistics.

## Contents

- [`01_ngrams_tutorial.py`](01_ngrams_tutorial.py) — runnable end-to-end tutorial. Trains unigram, bigram (by hand), and trigram (`MLE`, `Laplace`, `KneserNeyInterpolated` via `nltk.lm`) models on Jane Austen's *Emma* from NLTK's Gutenberg corpus. Generates sentences from each, evaluates with perplexity on a held-out test set.
- `unigram_zipf.png` — written by the script; the Zipf-law frequency plot of the top tokens.

## How to run

```bash
python 01_ngrams/01_ngrams_tutorial.py
```

The script self-bootstraps: it `pip install`s `nltk` and `matplotlib` if missing, downloads the small NLTK data packages (`gutenberg`, `punkt`, `punkt_tab`) on first run, then prints each section's output to stdout. Read the source top-to-bottom while watching the prints scroll by — every meaningful step is annotated with a *why* comment and a labelled print.

## Planned next

- Hand-rolled add-k smoothing with a `k` ablation.
- Stupid back-off as a contrast to interpolation.
- A short exercise file: train on Moby Dick instead, compare perplexity, discuss what changes.
- Discussion notes: what n-grams cannot represent (long-range dependencies, paraphrase, reference).

## Prerequisite

- Python basics from [`../00_python_intro/`](../00_python_intro/).

## Bridge to next unit

The tutorial closes by identifying the central limitation of n-grams — they cannot summarise arbitrary context. The next unit, [`../02_hmm/HMM.md`](../02_hmm/HMM.md), fixes this with *latent state*.
