# 🧬 Unit 3 — Contextual Embeddings

**Status:** ⬜ Planned

## Learning goal

Move from discrete counts (n-grams) and discrete latent states (HMMs) to *continuous vector representations* of words. Then push past static embeddings (one vector per word) to **contextual** embeddings, where the same word gets different vectors in different sentences.

## Planned contents

- Static embeddings: word2vec (skip-gram), GloVe — what "meaning as a vector" buys us.
- Distributional hypothesis: similar context ⇒ similar vector. Empirically check it.
- Why static embeddings fail: *bank* (river) vs. *bank* (finance) get the same vector.
- Contextual embeddings: ELMo / BERT-style — token representations that depend on the surrounding sequence.
- Visualizing embedding space (PCA / t-SNE / UMAP) and probing what dimensions encode.

## Prerequisite

- [`02_hmm/`](../02_hmm/) — for the contrast between discrete latent states and continuous representations.

## Bridge to next unit

This unit motivates [`04_transformers/`](../04_transformers/): contextual embeddings are what transformers *produce*, and self-attention is the mechanism that produces them.
