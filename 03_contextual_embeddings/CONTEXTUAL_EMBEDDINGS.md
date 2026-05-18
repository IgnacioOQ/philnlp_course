# 🧬 Unit 3 — Contextual Embeddings

**Status:** 🟡 In progress — Skip-Gram notebook done (static embeddings); contextual / ELMo / BERT section still planned.

## 📓 Notebooks

- [`Skip-Gram.ipynb`](Skip-Gram.ipynb) — from the distributional hypothesis to `king − man + woman ≈ queen`, organised in 8 numbered sections:
  - **§0 Setup** — Colab detection, deps, run-mode flags.
  - **§1 The skip-gram model** — distributional hypothesis (§1.1), architecture and forward pass (§1.2), cross-entropy loss and gradients (§1.3).
  - **§2 Toy implementation in pure NumPy** — the equations from §1 running on a 9-word corpus.
  - **§3 Why the toy can't scale** — subsampling (§3.1) and negative sampling (§3.2), with the SGNS loss derived.
  - **§4 Training on a real corpus** — NLTK **Brown + Reuters** (~2.3M alphabetic tokens) over 5 epochs of SGNS in pure NumPy.
  - **§5 Inspecting the learned vectors** — cosine nearest neighbours (§5.1), PCA scatter (§5.2), honest reckoning about what 2.3M tokens *can't* do (§5.3).
  - **§6 Pretrained GloVe and the analogy demo** — loading 6B-token GloVe (§6.1), running classic analogies as text (§6.2), and **visualising the parallelogram geometry** (§6.3) that makes the arithmetic work — gendered pairs and country↔capital pairs as side-by-side panels, plus a focused `{man, woman, king, queen}` parallelogram.
  - **§7 What static embeddings still can't do** — the bank/bank polysemy problem, bridge to Unit 4.
  - **§8 Cleanup** — optional Colab runtime disconnect.

### ☁️ Running on Google Colab

The notebook is Colab-ready and runs end-to-end without local setup:

1. Open [`Skip-Gram.ipynb`](Skip-Gram.ipynb) in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Run all.** The first cell ("Setup") auto-detects Colab and `pip install`s `gensim` (which Colab dropped from its base image around 2023). `nltk`, `scikit-learn`, `numpy`, `matplotlib` are pre-installed.
3. With the defaults (`SMOKE_TEST = False`, `AUTO_DISCONNECT = False`) the full notebook takes ~6–9 minutes on Colab's free CPU runtime: ~7 min for the SGNS training on Brown + Reuters (5 epochs, ~5.8M training pairs), plus a one-time ~134 MB download for the pretrained GloVe vectors. Local laptop CPU is similar (~8 min total measured on a 2024-era Mac).

**Two flags in the Setup cell are worth knowing:**

- `SMOKE_TEST = True` caps the Brown+Reuters corpus at 100k tokens and runs **1 epoch** of SGNS — the full notebook then finishes in well under a minute. Useful for iterating on downstream cells (PCA, analogies) without waiting on training each restart.
- `AUTO_DISCONNECT = True` makes the very last cell disconnect the Colab runtime once the notebook finishes — only matters on paid Colab plans where leaving the tab open keeps billing. Off by default.

> ⚠️ **Colab's filesystem is ephemeral.** Each new Colab session re-downloads Brown + Reuters (~10 MB combined via NLTK) and GloVe (~134 MB via `gensim.downloader`). Local laptops keep both caches in `~/nltk_data/` and `~/gensim-data/` between runs.

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
