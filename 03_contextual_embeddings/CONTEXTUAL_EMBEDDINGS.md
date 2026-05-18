# 🧬 Unit 3 — Contextual Embeddings

**Status:** 🟡 In progress — Skip-Gram notebook done (static embeddings); contextual / ELMo / BERT section still planned.

## 📓 Notebooks

- [`Skip-Gram.ipynb`](Skip-Gram.ipynb) — from the distributional hypothesis to `king − man + woman ≈ queen`. The full path:
  1. The skip-gram architecture and cross-entropy loss, derived from scratch.
  2. A pure-NumPy implementation on a toy corpus.
  3. The two tricks that make Word2Vec actually work — **subsampling** of frequent words and **negative sampling**.
  4. Training Skip-Gram with Negative Sampling on the **NLTK Brown corpus** (~1M tokens) in pure NumPy.
  5. Nearest-neighbour and PCA inspection of the learned vectors.
  6. Loading **pretrained GloVe** (Wikipedia + Gigaword, 6B tokens) and running the classic vector-arithmetic analogies.
  7. The **bank / bank** problem — what static embeddings cannot do, motivating Unit 4.

### ☁️ Running on Google Colab

The notebook is Colab-ready and runs end-to-end without local setup:

1. Open [`Skip-Gram.ipynb`](Skip-Gram.ipynb) in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Run all.** The first cell ("Setup") auto-detects Colab and `pip install`s `gensim` (which Colab dropped from its base image around 2023). `nltk`, `scikit-learn`, `numpy`, `matplotlib` are pre-installed.
3. With the defaults (`SMOKE_TEST = False`, `AUTO_DISCONNECT = False`) the full notebook takes ~3–5 minutes on Colab's free CPU runtime: ~150–200 s for the SGNS training on Brown, plus a one-time ~134 MB download for the pretrained GloVe vectors.

**Two flags in the Setup cell are worth knowing:**

- `SMOKE_TEST = True` caps the Brown corpus at 100k tokens and runs **1 epoch** of SGNS — the full notebook then finishes in well under a minute. Useful for iterating on downstream cells (PCA, analogies) without waiting on training each restart.
- `AUTO_DISCONNECT = True` makes the very last cell disconnect the Colab runtime once the notebook finishes — only matters on paid Colab plans where leaving the tab open keeps billing. Off by default.

> ⚠️ **Colab's filesystem is ephemeral.** Each new Colab session re-downloads Brown (~3 MB via NLTK) and GloVe (~134 MB via `gensim.downloader`). Local laptops keep both caches in `~/nltk_data/` and `~/gensim-data/` between runs.

## 📜 Source papers (in this folder)

- Mikolov et al. (2013a) — *Efficient Estimation of Word Representations in Vector Space* (`6. Estimations of representations.pdf`)
- Mikolov et al. (2013b) — *Distributed Representations of Words and Phrases and their Compositionality* (`6. Distributed represenattions.pdf`)

## 🎯 Learning goal

Move from discrete counts (n-grams) and discrete latent states (HMMs) to *continuous vector representations* of words. Then push past static embeddings (one vector per word) to **contextual** embeddings, where the same word gets different vectors in different sentences.

## 🗺️ Planned contents

- ✅ Static embeddings: Word2Vec skip-gram, GloVe — what "meaning as a vector" buys us.
- ✅ Distributional hypothesis: similar context ⇒ similar vector. Empirically check it.
- ✅ Why static embeddings fail: *bank* (river) vs. *bank* (finance) get the same vector.
- ⬜ Contextual embeddings: ELMo / BERT-style — token representations that depend on the surrounding sequence.
- ⬜ Visualising embedding space (t-SNE / UMAP) and probing what dimensions encode.

## 🔗 Prerequisite

- [`02_hmm/`](../02_hmm/) — for the contrast between discrete latent states and continuous representations.

## 🌉 Bridge to next unit

This unit motivates [`04_transformers/`](../04_transformers/): contextual embeddings are what transformers *produce*, and self-attention is the mechanism that produces them.
