# The Transformer Architecture: A Slow, Hands-On Tutorial
- status: active
- type: tutorial
- id: ml.transformer_tutorial
- description: Slow, hands-on walk through the transformer architecture, building self-attention from dot products up to a full multi-head encoder-decoder in PyTorch, with NumPy worked examples on real English sentences and attention-weight heatmap visualizations.
- label: [agent, source-material]
- injection: procedural
- volatility: initial_draft
- last_checked: 2026-04-26
<!-- content -->

This tutorial walks an LLM coding agent (Claude, Gemini, etc.) and a human learner through the transformer architecture from the ground up. It does not assume prior knowledge of attention, only basic linear algebra (vectors, matrices, dot products) and basic Python. Every formula appears first as code on a real example, then as math, in that order — math is just a notation for the code that produced the numbers.

The progression is deliberate and slow: tokenization → embeddings → dot-product similarity → softmax → single-head self-attention → queries/keys/values → scaling → multi-head attention → positional encodings → feed-forward sublayers → residual connections and layer normalization → encoder block → stacking → decoder and cross-attention → a complete transformer trained on real text. Every step has runnable Python code, and every conceptually important step ends with a small experiment on a real English sentence.

Reach for this tutorial when the goal is *understanding* — building the mental model of why each piece exists. For a quick reference of which class is in `torch.nn`, look elsewhere. For "give me a transformer that runs on my data tomorrow," the final section provides a working implementation, but the value of this document is the path that gets you there.

## 0. How to Use This Tutorial

This tutorial is meant to be read **with code running in front of you**. Every code block is self-contained at its level — early blocks use only NumPy so the math is fully visible; later blocks switch to PyTorch once we want gradients and GPU support.

**Setup.** You need Python 3.10 or newer and these packages:

```bash
pip install numpy matplotlib torch notebook
```

That is the entire dependency list. No HuggingFace, no transformers library — we are building the thing those libraries wrap.

**Each section of this tutorial must be implemented as a Jupyter notebook** (`.ipynb`), one notebook per section. Jupyter notebooks are required because they let you run code cell by cell, see output immediately below each cell, and mix explanatory text with live code — all of which are essential to the way this tutorial is structured. Do not use plain `.py` scripts.

**Notebooks must be numbered and ordered.** Name each file with a two-digit prefix matching the section number, followed by an underscore and a short descriptive name — for example `01_tokenizer.ipynb`, `02_embeddings.ipynb`, `03_attention.ipynb`. The numeric prefix ensures notebooks sort correctly in any file browser and makes the progression of the tutorial immediately visible. Never create a notebook without a numeric prefix.

**The pedagogical contract.** Each notebook follows the same shape:

1. A markdown cell stating *what we are about to build and why*.
2. A worked numerical example, usually on a tiny sentence like "the cat sat on the mat".
3. The code, with the following two requirements strictly enforced:
   - **Inline comments on every non-trivial line.** Comments must explain *why* the code does what it does, not just restate what the line does. For example, `# subtract max for numerical stability, not correctness` is a good comment; `# subtract max` is not.
   - **Print statements after every meaningful computation.** Every variable whose value would surprise a first-time reader must be printed, with a label. Shapes, intermediate values, and sanity checks (e.g. "row sums should be 1.0") must all appear as printed output so the learner can verify their mental model against the actual numbers.
4. The math, presented in a markdown cell as a summary of what the code did.
5. A visualization or printout that you should actually run and look at.

If you skip the "actually look at it" step, the rest of the tutorial gets harder. The whole point of attention is that it is *visualizable* — the matrix of attention weights tells you what the model is doing. Looking at those matrices is how the architecture stops being abstract.

**For the LLM agent reading this with a user.** When you reach a code block, implement it as a new cell in the current section's notebook. Every cell must satisfy the two requirements above (inline comments and print statements) before moving on. Run the cell, show the output, and let the user ask questions before proceeding to the next cell. The order matters. Do not jump ahead to multi-head attention before the user is comfortable with single-head; the multi-head version is just "do the single-head version several times in parallel" and that intuition only lands if single-head is solid first.

## 1. Why Transformers? What Problem Are We Solving?

Before any architecture, the problem. We have a sequence of tokens — words, subwords, characters, audio frames, image patches, anything ordered. We want a model that, given the sequence, produces a useful representation: a vector per token that captures not only what that token is in isolation but how it relates to the rest of the sequence.

Concretely, in the sentence

> The animal didn't cross the street because it was too tired.

the word **it** is ambiguous in isolation. To resolve it, the model needs the representation of "it" to depend on "animal" (its referent), not on every word equally. The model should *learn which other tokens are relevant for each token*, and use that selectivity to build context-aware representations.

Before transformers, the dominant tools for this were:

- **Recurrent neural networks (RNNs, LSTMs, GRUs)** — process the sequence left-to-right, carrying a hidden state. They work, but two problems: (a) the hidden state is a fixed-size bottleneck, so information from far back can get squeezed out, and (b) computation is inherently sequential, so they cannot exploit GPU parallelism over the sequence dimension.
- **Convolutions over sequences** — fast and parallel, but each layer only sees a fixed window. To capture long-range dependencies you need many layers, and even then the receptive field grows slowly.

The 2017 paper *Attention Is All You Need* (Vaswani et al.) made a simple structural claim: drop the recurrence and the convolutions, and use attention as the only mechanism for tokens to interact. The resulting architecture — the transformer — is fully parallel over the sequence (great for GPUs), gives every token direct access to every other token (no bottleneck), and turned out to scale much better with data and parameters than its predecessors. Every modern large language model (GPT, Claude, Gemini, Llama) is a descendant of that architecture.

Our destination is a working transformer. Our path goes through every piece, in dependency order: you cannot understand multi-head attention without scaled dot-product attention; you cannot understand scaled dot-product attention without softmax; you cannot understand softmax-based attention without dot-product similarity; and you cannot understand dot-product similarity without embeddings. So we start with embeddings.

## 2. From Text to Numbers: Tokenization

Neural networks operate on numbers. Text is symbols. The bridge between them is **tokenization**: a deterministic procedure that splits a string into a list of *tokens* and assigns each token an integer ID.

There are three common granularities:

- **Character-level** — every character is a token. Vocabulary is tiny (~100 for English including punctuation). Sequences are long. Easy to implement, hard for the model to learn meaning because individual characters carry little information.
- **Word-level** — every space-separated word is a token. Vocabulary is huge (hundreds of thousands) and the model has no graceful way to handle words it has never seen ("antidisestablishmentarianism", typos, new product names).
- **Subword (BPE, WordPiece, SentencePiece)** — the pragmatic compromise modern models use. Common words become single tokens; rare words are split into reusable pieces. Vocabulary stays moderate (30k–100k), and out-of-vocabulary words are handled gracefully by composition.

For this tutorial, character-level is enough. It keeps the vocabulary tiny, makes everything human-readable in printouts, and lets us focus on the architecture rather than the tokenizer. Real systems use BPE or SentencePiece; the architecture above the tokenizer is identical.

Let us build a character-level tokenizer:

```python
# tokenizer.py
# A character-level tokenizer: maps each unique character in the corpus
# to a unique integer ID and back. This is the simplest possible tokenizer;
# it has no notion of words or subwords, only characters.

class CharTokenizer:
    def __init__(self, corpus: str):
        # Find every unique character that appears in the training corpus.
        # `sorted` makes the mapping deterministic across runs — without it,
        # the IDs would change every time we re-instantiate the tokenizer.
        unique_chars = sorted(set(corpus))

        # Build the two lookup tables. `stoi` = string-to-integer,
        # `itos` = integer-to-string. Convention borrowed from Karpathy's nanoGPT.
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Vocabulary size: the embedding matrix later will need this.
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> list[int]:
        # Convert each character to its integer ID. If the input contains
        # a character not seen during construction, this will raise KeyError.
        # Real tokenizers handle unknowns with a special <unk> token; we don't
        # bother for this tutorial.
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        # Reverse direction: integer IDs back to characters, then join.
        return "".join(self.itos[i] for i in ids)


# Quick demonstration on our running example sentence.
if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)

    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Vocabulary: {sorted(tok.stoi.keys())}")

    ids = tok.encode(sentence)
    print(f"Encoded: {ids}")
    print(f"Decoded: {tok.decode(ids)!r}")
```

Running this prints something like:

```
Vocabulary size: 8
Vocabulary: [' ', 'a', 'c', 'e', 'h', 'm', 'n', 'o', 's', 't']
Encoded: [9, 4, 3, 0, 2, 1, 9, 0, 8, 1, 9, 0, 7, 6, 0, 9, 4, 3, 0, 5, 1, 9]
Decoded: 'the cat sat on the mat'
```

The decoded string matches the original — that is the round-trip invariant any correct tokenizer must satisfy. From here on, when we say "token", we mean an integer ID like `9` or `4`, not a character or a word. The model will never see the string "the"; it will see the sequence of integers `[9, 4, 3]`.

## 3. From Tokens to Vectors: Embeddings

The model still cannot work directly with integer IDs. The integer 9 ("t" in our example) is not 9× more important than the integer 1 (" "); the IDs are arbitrary labels with no numerical meaning. To give the model something it *can* compute with, we associate each token ID with a learned **vector** of real numbers — a dense embedding.

If our vocabulary has `V` tokens and we choose embedding dimension `d_model`, the **embedding matrix** is a `V × d_model` table. Looking up the embedding of token `i` is just selecting row `i`. Critically, the entries of this matrix are *parameters*: gradient descent will adjust them during training so that semantically similar tokens end up with geometrically similar vectors.

Geometric similarity matters because we will measure relationships between tokens using **dot products** of their embeddings. Two vectors with high dot product point in similar directions; two with low or negative dot product do not. So if "cat" and "dog" should be treated as similar by the model, training will pull their embedding vectors together; if "cat" and "Wednesday" should not, training will push them apart. This is the geometric content of "the model learns word meanings."

For now we are not training, only inspecting, so we initialize embeddings randomly. Random vectors will not encode any real semantics, but they let us trace the mechanics:

```python
# embeddings.py
import numpy as np

# Re-use the tokenizer from the previous section.
from tokenizer import CharTokenizer

# Hyperparameters. d_model is the dimensionality of every token vector
# throughout the model — it stays fixed end-to-end. The original paper used
# d_model = 512. We use 8 here so we can print full vectors and matrices.
D_MODEL = 8

# Set a seed so the random embeddings are reproducible across runs.
# Reproducibility is non-negotiable for debugging; without a seed,
# every print of "the embedding of 't'" would be different.
rng = np.random.default_rng(seed=42)


def make_embedding_matrix(vocab_size: int, d_model: int) -> np.ndarray:
    """
    Create a (vocab_size, d_model) matrix of random embeddings.

    We scale by 1/sqrt(d_model) to keep the typical magnitude of an
    embedding vector roughly O(1). Without this scaling, larger d_model
    would mean larger vector norms, which would in turn produce larger
    dot products downstream and saturate the softmax.
    """
    scale = 1.0 / np.sqrt(d_model)
    return rng.normal(loc=0.0, scale=scale, size=(vocab_size, d_model))


def embed(token_ids: list[int], embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Look up the embedding for each token ID. The result has shape
    (sequence_length, d_model): one row per token, in order.

    In NumPy, `embedding_matrix[token_ids]` performs fancy indexing:
    it selects the rows whose indices appear in `token_ids`, in order.
    """
    return embedding_matrix[token_ids]


if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)
    ids = tok.encode(sentence)

    E = make_embedding_matrix(tok.vocab_size, D_MODEL)
    X = embed(ids, E)

    print(f"Embedding matrix shape: {E.shape}  (vocab_size, d_model)")
    print(f"Sequence embedding shape: {X.shape}  (seq_len, d_model)")
    print()
    print("First three token embeddings (rows of X):")
    print(X[:3].round(3))
```

Running this gives a matrix `X` of shape `(22, 8)`: 22 characters in the sentence, 8 dimensions per character. From this point on, the entire transformer operates on `X` — a `(seq_len, d_model)` matrix of vectors. We will transform it many times, but we never go back to integers until the very end (when we map the final vectors to vocabulary logits to predict the next token).

A useful mental picture: imagine a cloud of points in 8-dimensional space. Each token is a point. Attention is going to be about *moving each point toward the points it considers relevant* — context-mixing in vector space.

## 4. The Heart of the Matter: Attention, Step by Step

This is the central section of the tutorial. Everything else is plumbing. We build attention in seven small steps, each of which is a one- or two-line code change on the previous step. By the end of section 4 you will have a complete, scaled, learnable self-attention function that you fully understand.

### 4.1 The Intuition: Weighted Averages

Recall the running example:

> The animal didn't cross the street because it was too tired.

For "it" to resolve to "animal", the vector representing "it" — call it `x_it` — should, after attention, be informed by the vector for "animal". One natural way to do this: replace `x_it` with a *weighted average* of all the token vectors in the sentence, where the weights are large for "animal" and small for everything else.

That is essentially what attention is. The wrinkle: we do not hard-code which tokens get high weight. We compute the weights *from the vectors themselves*, using a similarity function. Tokens whose vectors look "compatible" with `x_it` get high weight; the rest get low weight. The model learns the embeddings such that the right tokens end up looking compatible.

Concretely, attention takes a sequence `X` of shape `(seq_len, d_model)` and produces another sequence `Y` of the same shape, where each row of `Y` is a weighted average of rows of `X`. The weights for row `i` form a probability distribution over rows `0, 1, ..., seq_len - 1`. We just need a way to compute those weights.

### 4.2 Dot Product as Similarity

The standard similarity function for vectors is the dot product. For two vectors `u` and `v` in `R^d`,

```
u · v = u[0]*v[0] + u[1]*v[1] + ... + u[d-1]*v[d-1]
```

Geometrically, `u · v = |u| |v| cos(θ)` where `θ` is the angle between them. Two vectors pointing in the same direction have a large positive dot product; orthogonal vectors have zero; opposite vectors have a large negative dot product.

If we have a matrix `X` of shape `(seq_len, d_model)`, then `X @ X.T` is a `(seq_len, seq_len)` matrix where entry `(i, j)` is the dot product of token `i`'s vector with token `j`'s vector. This matrix is the **raw similarity matrix** of the sequence. It is the seed of attention.

```python
# similarity.py
import numpy as np
from tokenizer import CharTokenizer
from embeddings import make_embedding_matrix, embed, D_MODEL

sentence = "the cat sat on the mat"
tok = CharTokenizer(sentence)
ids = tok.encode(sentence)
E = make_embedding_matrix(tok.vocab_size, D_MODEL)
X = embed(ids, E)  # shape (22, 8)

# X @ X.T gives a (22, 22) matrix of raw dot-product similarities.
# Entry (i, j) is the dot product of the embedding of the i-th character
# with the embedding of the j-th character.
S = X @ X.T

# A useful sanity check: with random embeddings, characters that share an
# identity (e.g. all the 't's in "the cat sat") should have the highest
# self-similarity, because they share the same embedding row.
# Let's print one row to inspect.
print(f"Sentence positions and characters:")
for i, c in enumerate(sentence):
    print(f"  {i:2d}: {c!r}")

print()
print(f"Raw similarity row for position 0 (character {sentence[0]!r}):")
print(S[0].round(2))
```

The row `S[0]` tells us, in raw form, how much the first character ("t") "resembles" each other character. Positions where the same character appears (other "t"s in "the cat sat") will have high similarity because they look up the same embedding row. This is a degenerate case driven by character identity, but the idea generalizes: when embeddings are trained, semantic similarity replaces identity, and `X @ X.T` becomes a meaningful relevance matrix.

There is a problem though: the raw similarities are unbounded real numbers — some positive, some negative, no constraint. We want *weights* that sum to 1 (a proper weighted average). For that, we need softmax.

### 4.3 Softmax: Turning Scores into Weights

The softmax function turns a vector of arbitrary real numbers into a probability distribution. For a vector `s = [s_1, ..., s_n]`,

```
softmax(s)_i = exp(s_i) / sum_j(exp(s_j))
```

Three properties matter:

1. **Non-negative**: `exp` is always positive.
2. **Sums to 1**: that is what the denominator enforces.
3. **Sharpening**: because of the exponential, larger inputs become disproportionately larger outputs. A score of 5 versus 4 produces weights of roughly 0.73 versus 0.27 — softmax magnifies differences.

One numerical subtlety: `exp(s_i)` overflows for large `s_i`. The standard fix is to subtract the maximum first: `softmax(s) = softmax(s - max(s))`. This is mathematically identical (the constant cancels in numerator and denominator) but numerically stable.

```python
# softmax.py
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax along a given axis.

    Mathematically: softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
    Implementation: subtract the per-row maximum before exponentiating,
    so the largest exponent is exp(0) = 1 and there is no overflow.

    Parameters
    ----------
    x : np.ndarray
        Input array of arbitrary shape.
    axis : int
        Axis along which softmax is computed. Default -1 (last axis),
        which is the standard convention for sequence-of-vectors inputs.

    Returns
    -------
    np.ndarray
        Same shape as input; values along `axis` are non-negative and sum to 1.
    """
    # Subtract the max along `axis`, keeping dims so broadcasting works.
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    # Exponentiate.
    exp_x = np.exp(x_shifted)
    # Normalize so the values along `axis` sum to 1.
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


if __name__ == "__main__":
    # Demonstrate the sharpening effect.
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    print("Raw scores:           ", raw)
    print("After softmax:        ", softmax(raw).round(3))

    # Bigger scale, sharper distribution: the difference of 1 between
    # adjacent scores is now relatively larger.
    print("Raw scores * 5:       ", raw * 5)
    print("After softmax (×5):   ", softmax(raw * 5).round(3))

    # The opposite extreme: very small differences yield near-uniform weights.
    print("Raw scores * 0.1:     ", raw * 0.1)
    print("After softmax (×0.1): ", softmax(raw * 0.1).round(3))
```

The `×5` and `×0.1` cases show why scaling matters: the same *pattern* of scores produces a peaked distribution at large scale and a nearly uniform distribution at small scale. We will return to this in section 4.6 when we talk about the `√d_k` divisor — that divisor exists precisely to prevent the "too peaked" regime.

### 4.4 The First Attention: Self-Attention as Weighted Average

We now have everything to write the simplest possible self-attention: similarities → softmax → weighted average.

```python
# self_attention_v0.py
# Version 0 of self-attention: no learnable parameters yet, just the mechanism.
# Each output row is a softmax-weighted average of input rows, where the
# weights come from raw dot-product similarities of input rows with each other.

import numpy as np
from tokenizer import CharTokenizer
from embeddings import make_embedding_matrix, embed, D_MODEL
from softmax import softmax


def naive_self_attention(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simplest self-attention: similarity = dot product, weights = softmax,
    output = weighted average of input rows.

    Parameters
    ----------
    X : np.ndarray, shape (seq_len, d_model)
        Sequence of token embeddings.

    Returns
    -------
    Y : np.ndarray, shape (seq_len, d_model)
        Context-mixed embeddings. Y[i] is a weighted average over X.
    A : np.ndarray, shape (seq_len, seq_len)
        Attention weights. A[i, j] is the weight token i places on token j.
        Each row sums to 1.
    """
    # Step 1: compute raw similarity scores.
    # X has shape (T, d). X.T has shape (d, T). The product has shape (T, T).
    # scores[i, j] = X[i] . X[j], the dot product of token i and token j.
    scores = X @ X.T  # shape (T, T)

    # Step 2: convert each row of scores into a probability distribution.
    # We softmax along axis=-1 so that A[i, :] sums to 1 — token i's
    # attention is normalized over all tokens it can attend to.
    A = softmax(scores, axis=-1)  # shape (T, T)

    # Step 3: produce each output row as a weighted average of input rows.
    # A is (T, T), X is (T, d). A @ X has shape (T, d):
    #   (A @ X)[i, k] = sum_j A[i, j] * X[j, k]
    # which is exactly "row i of the output is a weighted sum of rows of X
    # with weights A[i, :]".
    Y = A @ X  # shape (T, d)

    return Y, A


if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)
    ids = tok.encode(sentence)
    E = make_embedding_matrix(tok.vocab_size, D_MODEL)
    X = embed(ids, E)

    Y, A = naive_self_attention(X)

    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {Y.shape}")
    print(f"Attention shape: {A.shape}")
    print()

    # Sanity: every row of A sums to 1.
    print(f"Row sums of A (should all be 1.0): {A.sum(axis=-1).round(4)}")
    print()

    # Print attention from the first 't' to every other position.
    print(f"Attention from position 0 ({sentence[0]!r}) to all positions:")
    for j, c in enumerate(sentence):
        bar = "#" * int(A[0, j] * 100)
        print(f"  pos {j:2d} ({c!r}): {A[0, j]:.3f}  {bar}")
```

Run this and look at the printout. With random embeddings, the highest attention from position 0 ("t") will go to other positions whose character embedding happens to be close to "t"'s embedding — most strongly to the other "t"s in the sentence (positions 11 and so on). That is the mechanism *working as designed*; whether it is *useful* depends on whether the embeddings encode useful relationships, which only happens after training.

This is real attention. Everything from here is refinement — making it more expressive, more stable, more parallelizable. But the core mechanism is what you just wrote: similarity, softmax, weighted average.

### 4.5 Queries, Keys, Values: the Three Roles

Naive self-attention has a structural limitation: the same vector `X[i]` plays *three different roles* in the formula:

1. The thing whose neighbors we are scoring — the **query**.
2. The thing being scored against — the **key**.
3. The thing whose content gets averaged — the **value**.

These three roles are conceptually different. When token "it" looks for what to attend to, the question it asks ("what am I looking for?") is not necessarily the same as how it advertises itself to others ("what do I offer for matching?"), and neither is the same as what it contributes to the output if matched ("what content do I add?"). Tying all three to the same vector limits the model's expressiveness.

The fix: introduce three learned projections. Three weight matrices `W_Q`, `W_K`, `W_V`, each of shape `(d_model, d_k)` (typically `d_k = d_model`), turn `X` into three different views:

```
Q = X @ W_Q   # what each token is querying for
K = X @ W_K   # what each token offers for matching
V = X @ W_V   # what each token contributes if matched
```

Now the attention formula becomes

```
A = softmax(Q @ K.T)
Y = A @ V
```

— similarities are between queries and keys, but the weighted average is over values. The three projections decouple the three roles and let the model learn distinct functions for each.

A useful analogy: think of `Q @ K.T` as a soft, differentiable database lookup. The query is your search string; each key is the title of an entry; the dot product `Q[i] @ K[j].T` is how well entry `j` matches your search. Softmax turns the scores into a soft selection. The value `V[j]` is the content of entry `j`. The output is the soft-selected content.

```python
# self_attention_v1.py
# Version 1: introduce learned Q, K, V projections.

import numpy as np
from tokenizer import CharTokenizer
from embeddings import make_embedding_matrix, embed, D_MODEL
from softmax import softmax


# d_k is the dimensionality of queries and keys. d_v is the dim of values.
# In the original transformer, d_k = d_v = d_model / n_heads. With one head,
# the simplest choice is d_k = d_v = d_model.
D_K = D_MODEL
D_V = D_MODEL


def make_attention_weights(d_model: int, d_k: int, d_v: int, rng) -> dict:
    """
    Initialize the three projection matrices for one attention head.

    Returns a dict keyed by 'W_Q', 'W_K', 'W_V'. Using a dict (rather than
    positional return) makes downstream code self-documenting.
    """
    # Same scaling logic as embeddings: keep activation magnitudes ~ O(1).
    scale = 1.0 / np.sqrt(d_model)
    return {
        "W_Q": rng.normal(0.0, scale, size=(d_model, d_k)),
        "W_K": rng.normal(0.0, scale, size=(d_model, d_k)),
        "W_V": rng.normal(0.0, scale, size=(d_model, d_v)),
    }


def qkv_self_attention(X: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Self-attention with separate learned Q, K, V projections.

    Parameters
    ----------
    X : (T, d_model) input sequence.
    params : dict with keys 'W_Q', 'W_K', 'W_V' for the projection matrices.

    Returns
    -------
    Y : (T, d_v) attention output.
    A : (T, T) attention weights.
    """
    # Project the same input three ways. Each row of Q, K, V corresponds
    # to the same token but expresses a different "view" of it.
    Q = X @ params["W_Q"]  # (T, d_k)
    K = X @ params["W_K"]  # (T, d_k)
    V = X @ params["W_V"]  # (T, d_v)

    # Score each query against every key.
    # Q is (T, d_k), K.T is (d_k, T), so Q @ K.T is (T, T).
    # scores[i, j] = Q[i] . K[j] — how well token i's query matches token j's key.
    scores = Q @ K.T  # (T, T)

    # Softmax over keys (axis=-1) so each query distributes attention over
    # all keys, summing to 1. This row-wise softmax means every token has
    # exactly one unit of attention to spend across the sequence.
    A = softmax(scores, axis=-1)  # (T, T)

    # Weighted average of values. (T, T) @ (T, d_v) = (T, d_v).
    Y = A @ V  # (T, d_v)

    return Y, A


if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)
    ids = tok.encode(sentence)
    rng = np.random.default_rng(seed=42)
    E = make_embedding_matrix(tok.vocab_size, D_MODEL)
    X = embed(ids, E)

    params = make_attention_weights(D_MODEL, D_K, D_V, rng)
    Y, A = qkv_self_attention(X, params)

    print(f"Q, K, V conceptually decouple the three roles each token plays.")
    print(f"Output shape: {Y.shape}, attention shape: {A.shape}")
    print(f"Attention matrix row sums (should be 1): {A.sum(-1).round(4)}")
```

The shape contract is unchanged from v0: input `(T, d_model)`, output `(T, d_v)`, attention `(T, T)`. What changed is *what the model can learn*. With v0, the only learnable parameters were the embeddings — the attention pattern was mechanically dictated by them. With v1, the projections `W_Q`, `W_K`, `W_V` give the model three independent dials for shaping each of the three roles. This is what attention looks like in real transformers.

### 4.6 Scaled Dot-Product Attention

There is one final adjustment. As `d_k` grows, the dot products `Q[i] @ K[j].T` grow in magnitude — they are sums of `d_k` products, and the variance of a sum scales linearly with the number of terms. Concretely, if entries of `Q` and `K` have variance `σ²`, then dot products have variance `d_k · σ⁴`, so their typical magnitude scales like `√d_k`.

Why is that bad? Recall from section 4.3 that softmax of large-magnitude scores produces *very peaked* distributions. If most scores are around 10, one slightly larger score (say, 12) gets nearly all the weight; everything else gets weight near zero. That has two failure modes:

1. **Information loss**: attention collapses to "look at one token, ignore the rest". The model loses its ability to mix.
2. **Dead gradients**: where softmax is saturated, its gradient is essentially zero, so the model cannot learn to adjust the attention pattern. Training stalls.

The fix is to divide the scores by `√d_k` before softmaxing. This brings the typical magnitude back to O(1) regardless of `d_k`, keeping the softmax in its expressive, gradient-friendly regime. The full formula is

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

This is the **scaled dot-product attention** of the *Attention Is All You Need* paper. Memorize it — it appears in every transformer.

```python
# scaled_attention.py
import numpy as np
from softmax import softmax


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention, the canonical formulation.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Parameters
    ----------
    Q : (T_q, d_k) query matrix.
    K : (T_k, d_k) key matrix.
    V : (T_k, d_v) value matrix. Must have the same first dim as K because
        each key K[j] corresponds to value V[j].
    mask : optional (T_q, T_k) boolean array. Where mask is True, attention
        is suppressed. Used for causal masking (decoder), padding masking, etc.

    Returns
    -------
    Y : (T_q, d_v) output.
    A : (T_q, T_k) attention weights.
    """
    d_k = Q.shape[-1]

    # Raw scores. Same as before.
    scores = Q @ K.T  # (T_q, T_k)

    # The crucial scaling. Divide by sqrt(d_k) to keep the typical
    # magnitude of scores around O(1) regardless of d_k. This is THE line
    # that distinguishes the "scaled" version from the unscaled one.
    scores = scores / np.sqrt(d_k)

    # Apply mask if provided. Setting masked positions to -infinity ensures
    # that after softmax, those positions get weight exactly 0. We use a
    # very large negative number rather than -np.inf to avoid NaN in case
    # an entire row is masked (which would otherwise produce 0/0).
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax row-wise: each query distributes 1 unit of attention over keys.
    A = softmax(scores, axis=-1)

    # Weighted average of values.
    Y = A @ V

    return Y, A
```

This is the core attention primitive of every transformer. From here on, we will compose it (multi-head, cross-attention, masked) but never modify it.

### 4.7 Visualizing Attention

Attention is a `(seq_len, seq_len)` matrix of weights. The natural way to look at it is a heatmap where row `i` is the attention pattern of token `i`: bright cells show what token `i` attends to. Interpretability of transformers is largely about staring at these heatmaps.

```python
# visualize_attention.py
# Plot the attention matrix as a heatmap with token labels on both axes.
# This is the workhorse visualization for any transformer-related debugging.

import numpy as np
import matplotlib.pyplot as plt

from tokenizer import CharTokenizer
from embeddings import make_embedding_matrix, embed, D_MODEL
from self_attention_v1 import make_attention_weights, D_K, D_V
from scaled_attention import scaled_dot_product_attention


def plot_attention(A: np.ndarray, tokens: list[str], title: str = "Attention", save_path: str | None = None):
    """
    Render an attention matrix as a heatmap.

    Parameters
    ----------
    A : (T_q, T_k) attention weights. Each row should sum to 1.
    tokens : the T tokens, used to label both axes.
    title : figure title.
    save_path : if given, save the figure to this file. Otherwise plt.show().
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # imshow renders the matrix; lower attention is darker, higher is brighter.
    im = ax.imshow(A, cmap="viridis", aspect="auto")

    # Tick labels: queries on the y-axis (which token is "looking"),
    # keys on the x-axis (which token is "being looked at").
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key (token attended to)")
    ax.set_ylabel("Query (token attending)")
    ax.set_title(title)

    # Colorbar makes the magnitude scale legible.
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)
    ids = tok.encode(sentence)

    rng = np.random.default_rng(seed=42)
    E = make_embedding_matrix(tok.vocab_size, D_MODEL)
    X = embed(ids, E)
    params = make_attention_weights(D_MODEL, D_K, D_V, rng)

    # Use the scaled attention primitive directly with the QKV projections.
    Q = X @ params["W_Q"]
    K = X @ params["W_K"]
    V = X @ params["W_V"]
    Y, A = scaled_dot_product_attention(Q, K, V)

    # Token labels: just the characters, with a marker for spaces.
    tokens = [c if c != " " else "_" for c in sentence]
    plot_attention(A, tokens, title="Self-attention (random init)", save_path="attention_random.png")
    print("Saved attention_random.png")
```

What does the printout/heatmap look like with random parameters? Mostly noise. With random `W_Q` and `W_K`, the attention pattern carries no semantic structure — that is exactly what we should expect *before* training. The point of looking at it now is to confirm the mechanism is well-defined: rows sum to 1, dimensions match, the heatmap renders. After we train an actual model in section 13, we will revisit this and see the attention pattern become non-random in interpretable ways.

For pedagogical purposes here is what an idealized attention heatmap *looks like* in a trained model on the sentence "The animal didn't cross the street because it was too tired":

```
         The animal didn't cross the street because it  was  too  tired
The    [ .8   .05    .02    .01    .03  .03    .01    .02 .01  .01  .01]
animal [ .05  .80    .03    .02    .02  .02    .01    .03 .01  .01  .00]
didn't [ .02  .03    .80    .05    .02  .03    .02    .01 .01  .01  .00]
cross  [ .01  .02    .03    .82    .02  .07    .01    .01 .00  .01  .00]
the    [ .04  .02    .02    .03    .80  .05    .01    .02 .00  .01  .00]
street [ .02  .03    .02    .07    .05  .77    .01    .01 .01  .01  .00]
because[ .01  .02    .02    .01    .01  .02    .85    .03 .01  .01  .01]
it     [ .01  .55    .02    .01    .02  .12    .03    .15 .03  .03  .03]   <-- LOOK HERE
was    [ .01  .02    .01    .01    .01  .02    .03    .04 .80  .04  .01]
too    [ .01  .02    .01    .01    .01  .01    .01    .02 .04  .80  .07]
tired  [ .01  .35    .02    .01    .02  .04    .01    .15 .03  .07  .30]   <-- AND HERE
```

The interesting rows are "it" and "tired". "It" puts about 55% of its attention on "animal" — coreference resolution, in the weights. "Tired" puts roughly 35% on "animal" (the subject of "tired") and 30% on itself (most tokens have non-trivial self-attention because their own representation is informative). These are the kinds of patterns that emerge from training and that have made attention famous.

We have now finished the core. The rest of the architecture is composition, regularization, and depth. None of it is conceptually as deep as what you just read.

## 5. Multi-Head Attention

A single attention head produces *one* attention pattern per token: one weighted average over the sequence. But intuitively, each token relates to others in *multiple* ways simultaneously. "It" might want to track its referent (a coreference relationship) *and* its grammatical role (a syntactic relationship) *and* its semantic field (a topical relationship). A single attention head, having one set of `W_Q, W_K, W_V`, can only learn one composite of these.

**Multi-head attention** runs several attention heads in parallel, each with its own `W_Q, W_K, W_V`, and concatenates the outputs. With `h` heads of dimensionality `d_k = d_v = d_model / h`, the total parameter count and computation are roughly the same as one big head of dimension `d_model`, but the model gets `h` independent attention patterns per layer.

The procedure is:

1. Project `X` to `Q, K, V` of shape `(T, d_model)`, then split into `h` heads each of shape `(T, d_k)` where `d_k = d_model / h`.
2. Run scaled dot-product attention independently on each head.
3. Concatenate the `h` outputs along the feature dimension, recovering shape `(T, d_model)`.
4. Apply a final linear projection `W_O` of shape `(d_model, d_model)` to mix information across heads.

Step 4 is important: without `W_O`, the heads' outputs would be concatenated but never recombined, and downstream layers would have to do this mixing themselves. `W_O` makes the multi-head module a clean drop-in replacement for single-head attention.

```python
# multi_head_attention.py
import numpy as np
from scaled_attention import scaled_dot_product_attention


def make_multi_head_params(d_model: int, n_heads: int, rng) -> dict:
    """
    Initialize parameters for multi-head attention.

    We store W_Q, W_K, W_V each as a single (d_model, d_model) matrix
    rather than h separate (d_model, d_k) matrices. The split into heads
    happens by reshape in the forward pass. This is more efficient and
    maps directly to what PyTorch's nn.MultiheadAttention does internally.
    """
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    scale = 1.0 / np.sqrt(d_model)
    return {
        "W_Q": rng.normal(0.0, scale, size=(d_model, d_model)),
        "W_K": rng.normal(0.0, scale, size=(d_model, d_model)),
        "W_V": rng.normal(0.0, scale, size=(d_model, d_model)),
        "W_O": rng.normal(0.0, scale, size=(d_model, d_model)),
        "n_heads": n_heads,
    }


def multi_head_attention(X: np.ndarray, params: dict, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-head self-attention.

    Parameters
    ----------
    X : (T, d_model) input.
    params : dict from `make_multi_head_params`.
    mask : optional (T, T) mask, applied identically to every head.

    Returns
    -------
    Y : (T, d_model) output.
    A_all : (n_heads, T, T) per-head attention weights, useful for visualization.
    """
    T, d_model = X.shape
    h = params["n_heads"]
    d_k = d_model // h  # per-head dimensionality

    # Project X to Q, K, V, each of shape (T, d_model).
    Q = X @ params["W_Q"]
    K = X @ params["W_K"]
    V = X @ params["W_V"]

    # Reshape into (h, T, d_k) by splitting the feature dimension into h chunks.
    # The reshape goes (T, d_model) -> (T, h, d_k) -> (h, T, d_k) via transpose.
    # This rearrangement places the head axis first so we can iterate or
    # broadcast over heads cleanly.
    def split_heads(M):
        return M.reshape(T, h, d_k).transpose(1, 0, 2)

    Q_h = split_heads(Q)  # (h, T, d_k)
    K_h = split_heads(K)  # (h, T, d_k)
    V_h = split_heads(V)  # (h, T, d_k)

    # Run scaled dot-product attention per head. We collect the per-head
    # outputs and the per-head attention matrices.
    head_outputs = []
    head_attentions = []
    for i in range(h):
        Y_i, A_i = scaled_dot_product_attention(Q_h[i], K_h[i], V_h[i], mask=mask)
        head_outputs.append(Y_i)         # (T, d_k)
        head_attentions.append(A_i)      # (T, T)

    # Concatenate head outputs along the feature axis, restoring (T, d_model).
    # head_outputs is a list of h arrays, each (T, d_k). Stacking gives
    # (h, T, d_k); transposing to (T, h, d_k) and reshaping to (T, h*d_k) = (T, d_model)
    # produces the concatenation in the canonical "head 0's dims, then head 1's dims, ..." order.
    concat = np.stack(head_outputs, axis=0).transpose(1, 0, 2).reshape(T, d_model)

    # Final output projection. This mixes information across heads.
    Y = concat @ params["W_O"]

    A_all = np.stack(head_attentions, axis=0)  # (h, T, T)
    return Y, A_all


if __name__ == "__main__":
    from tokenizer import CharTokenizer
    from embeddings import make_embedding_matrix, embed, D_MODEL

    sentence = "the cat sat on the mat"
    tok = CharTokenizer(sentence)
    ids = tok.encode(sentence)

    rng = np.random.default_rng(seed=42)
    E = make_embedding_matrix(tok.vocab_size, D_MODEL)
    X = embed(ids, E)

    params = make_multi_head_params(D_MODEL, n_heads=2, rng=rng)
    Y, A_all = multi_head_attention(X, params)

    print(f"Input:  {X.shape}")
    print(f"Output: {Y.shape}  (same as input; multi-head is shape-preserving)")
    print(f"Per-head attentions: {A_all.shape}  (n_heads, T, T)")
    print()
    # Print the row of the first 't' for both heads to see they differ.
    print(f"Head 0 attention from position 0: {A_all[0, 0].round(2)}")
    print(f"Head 1 attention from position 0: {A_all[1, 0].round(2)}")
```

The two heads produce *different* attention patterns from the same input, because they have independent `W_Q, W_K, W_V` projections. In a trained model, head 0 might specialize in tracking subject-verb agreement while head 1 tracks long-range coreference; the Anthropic-style mechanistic-interpretability research has found heads with surprisingly clean specializations like "previous-token head", "induction head", "name-mover head". For our purposes the takeaway is that *more heads = more parallel relationships per layer*, and standard transformer configurations use 8–16 heads.

## 6. Position Matters: Positional Encodings

Self-attention has a property that is mathematically elegant and practically catastrophic: it is **permutation-equivariant**. Permute the input tokens and the output is permuted the same way — the model has no notion of order. Concretely, if you feed it "the cat sat on the mat" and "mat the on sat cat the", and align positions correctly, the per-token outputs are identical permutations of each other.

For language, this is a disaster. "Dog bites man" and "man bites dog" must produce different representations. We need to inject position information into the input.

The original transformer paper uses **sinusoidal positional encodings**. A `(max_len, d_model)` matrix where the entry at position `pos` and dimension `i` is

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

The wavelengths form a geometric series from `2π` to roughly `2π · 10000`, giving the model frequencies at every scale from "alternates every position" to "barely changes across a sequence of 10000 tokens". This particular form has two practical properties:

1. **Generalization to unseen lengths**: the formula is defined for any `pos`, so a model trained on sequences of length 512 can be evaluated on length 1024 without retraining the position embeddings.
2. **Linear relative positions**: there is a fixed linear transformation that takes `PE[pos]` to `PE[pos + k]` for any offset `k`. The model can in principle use this to compute relative positions cheaply.

In practice, modern transformers use **learned positional embeddings** (a `(max_len, d_model)` parameter matrix initialized randomly) or relative-position schemes like RoPE. For pedagogy, sinusoidal encodings are clearer because they are deterministic and visualizable.

```python
# positional_encoding.py
import numpy as np
import matplotlib.pyplot as plt


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Generate the canonical sinusoidal positional encoding matrix.

    Parameters
    ----------
    max_len : maximum sequence length the encoding will support.
    d_model : embedding dimensionality (must match the token embedding dim).

    Returns
    -------
    PE : (max_len, d_model) array. PE[pos] is the encoding for position `pos`.
    """
    # Allocate the output.
    PE = np.zeros((max_len, d_model), dtype=np.float32)

    # Position indices, shape (max_len, 1) for broadcasting with dim indices.
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)

    # Compute the inverse frequencies for even dimensions.
    # The original paper uses 10000^(2i/d_model), so the divisor exponent
    # is 2i/d_model and we get the *frequency* as 1 / 10000^(2i/d_model).
    # We compute it via exp(log(...)) for numerical stability and clarity.
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )  # (d_model/2,)

    # Even dimensions: sin. Odd dimensions: cos.
    # `position * div_term` broadcasts to (max_len, d_model/2).
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE


if __name__ == "__main__":
    # Visualize the positional encoding matrix.
    PE = sinusoidal_positional_encoding(max_len=64, d_model=64)
    print(f"PE shape: {PE.shape}")
    print(f"PE[0]: {PE[0, :8].round(3)}  (note sin(0)=0, cos(0)=1 alternating)")

    # Heatmap. Each row is a position; each column is a dimension. The
    # characteristic look is vertical bands of varying frequencies — high
    # frequencies on the left, low frequencies on the right.
    plt.figure(figsize=(10, 6))
    plt.imshow(PE, aspect="auto", cmap="RdBu")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Positional Encoding")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=120)
    plt.close()
    print("Saved positional_encoding.png")
```

The encoding is **added** to the token embeddings, not concatenated:

```python
X_with_pos = X + PE[:X.shape[0]]
```

Adding rather than concatenating keeps the dimensionality at `d_model` and lets the model decide, per dimension, how much to weight content vs. position. This works because positional encodings live in a different "subspace" of the embedding space than typical token vectors do, and the model learns to disentangle them.

## 7. The Other Sublayer: Position-wise Feed-Forward Networks

Each transformer layer has *two* sublayers: multi-head attention, which we just built, and a **position-wise feed-forward network (FFN)**, which we are about to build.

The FFN is intentionally simple — two linear layers with a non-linearity in between, applied identically to every token position:

```
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
```

(The non-linearity in the original paper is ReLU; modern variants use GELU or SwiGLU.)

Two things to notice:

1. **Position-wise**: the FFN is applied independently to each of the `T` token vectors. There is no mixing across positions in this sublayer — that is exclusively attention's job.
2. **Hidden expansion**: typically `W_1: (d_model, d_ff)` with `d_ff = 4 * d_model`, then `W_2: (d_ff, d_model)`. The hidden dimension is much larger than the input/output, which lets the FFN store a lot of "factual content" per token. Recent interpretability work suggests transformer FFNs act as key-value memories, with `W_1` selecting concepts and `W_2` retrieving associated content.

Why does the architecture need the FFN at all if attention already mixes information? Because attention is *linear* in the values: every output is a weighted sum of value vectors, with weights from softmax. Without a non-linearity, stacking attention layers would just give another linear function, and the model would be dramatically underpowered. The FFN's non-linearity (ReLU or GELU) is what lets the model build up genuinely non-linear representations across layers.

```python
# feed_forward.py
import numpy as np


def make_ffn_params(d_model: int, d_ff: int, rng) -> dict:
    """
    Allocate parameters for a position-wise feed-forward network.
    Standard sizing: d_ff = 4 * d_model.
    """
    scale_1 = 1.0 / np.sqrt(d_model)
    scale_2 = 1.0 / np.sqrt(d_ff)
    return {
        "W_1": rng.normal(0.0, scale_1, size=(d_model, d_ff)),
        "b_1": np.zeros(d_ff),
        "W_2": rng.normal(0.0, scale_2, size=(d_ff, d_model)),
        "b_2": np.zeros(d_model),
    }


def feed_forward(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Position-wise FFN: applied independently to every token vector.

    Parameters
    ----------
    X : (T, d_model)
    params : dict from `make_ffn_params`.

    Returns
    -------
    (T, d_model) — same shape as input.
    """
    # First linear layer: (T, d_model) @ (d_model, d_ff) = (T, d_ff).
    hidden = X @ params["W_1"] + params["b_1"]

    # ReLU non-linearity. We use np.maximum(0, hidden) which works elementwise.
    # The choice of ReLU vs GELU vs SwiGLU is empirical; ReLU is simplest and
    # what the original paper used.
    hidden = np.maximum(0.0, hidden)

    # Second linear layer projects back down to (T, d_model).
    out = hidden @ params["W_2"] + params["b_2"]

    return out
```

## 8. Residual Connections and Layer Normalization

Stacking many layers — and we want to stack many — runs into the classic deep-learning problem: gradients vanish or explode, training becomes unstable, and the model fails to learn. The transformer addresses this with two well-known tools.

**Residual connections** (He et al., 2015). Around each sublayer, the input is added to the output:

```
output = sublayer(input) + input
```

This `+ input` is the *residual* (or *skip*) connection. It has two virtues:

1. The sublayer only has to learn the *difference* from the identity. If a layer is not yet useful, it can output near-zero and the network behaves as if the layer were absent. This makes deep networks trainable from scratch — early in training, all sublayers behave as approximate identities and learn refinements over time.
2. Gradients have a direct path from the loss back to early layers via the `+ input` term, regardless of how the sublayer transforms them. Vanishing gradients become much less of a problem.

**Layer normalization** (Ba et al., 2016). After the residual sum, the result is normalized so that each token vector has mean 0 and variance 1 across its `d_model` features, then rescaled by learned per-feature parameters `γ` (gain) and `β` (bias):

```
LN(x) = γ * (x - mean(x)) / sqrt(var(x) + ε) + β
```

The mean and variance are taken across the feature dimension *of a single token vector*, not across the batch. This is what makes it "Layer" norm rather than "Batch" norm — it does not depend on the batch composition, so it works identically at training and inference time, and identically for sequences of any length.

There is a placement choice: **post-norm** (original paper) puts LayerNorm after the residual sum, **pre-norm** (modern preference) puts LayerNorm before the sublayer:

```
post-norm:  x + LN(sublayer(x))           # original
pre-norm:   x + sublayer(LN(x))           # modern
```

Pre-norm is more stable to train for deep models because the residual path is unnormalized — gradients flow back without distortion. Most modern transformers (GPT-2 onwards, Llama, etc.) use pre-norm. We will use pre-norm for the rest of this tutorial.

```python
# layer_norm.py
import numpy as np


def make_layer_norm_params(d_model: int) -> dict:
    """
    LayerNorm has two learnable parameters per feature: gain (gamma)
    and bias (beta). They start as identity (gain=1, bias=0).
    """
    return {
        "gamma": np.ones(d_model),
        "beta": np.zeros(d_model),
    }


def layer_norm(X: np.ndarray, params: dict, eps: float = 1e-5) -> np.ndarray:
    """
    Layer normalization across the feature dimension.

    Parameters
    ----------
    X : (..., d_model) — any leading shape, last axis is features.
    params : dict with 'gamma' and 'beta' both of shape (d_model,).
    eps : small constant for numerical stability when variance is small.

    Returns
    -------
    Same shape as X. Each (d_model,)-vector is normalized to mean 0,
    variance 1, then rescaled and shifted by gamma and beta.
    """
    # Compute mean and variance across the last axis (features).
    # keepdims=True so the result broadcasts cleanly with X.
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)

    # Normalize.
    X_normed = (X - mean) / np.sqrt(var + eps)

    # Rescale and shift.
    return params["gamma"] * X_normed + params["beta"]
```

## 9. The Encoder Block

We can now assemble the standard encoder block. With pre-norm and the components from sections 5–8:

```
def encoder_block(x):
    a = layer_norm(x)
    a = multi_head_attention(a)
    x = x + a                      # first residual

    f = layer_norm(x)
    f = feed_forward(f)
    x = x + f                      # second residual

    return x
```

That is the full block. Two sublayers, each wrapped in a residual + LayerNorm pattern. Shape is preserved end-to-end: `(T, d_model)` in, `(T, d_model)` out.

```python
# encoder_block.py
import numpy as np

from multi_head_attention import make_multi_head_params, multi_head_attention
from feed_forward import make_ffn_params, feed_forward
from layer_norm import make_layer_norm_params, layer_norm


def make_encoder_block_params(d_model: int, n_heads: int, d_ff: int, rng) -> dict:
    """
    Bundle all parameters for one encoder block: MHA, FFN, two LayerNorms.
    """
    return {
        "mha":  make_multi_head_params(d_model, n_heads, rng),
        "ffn":  make_ffn_params(d_model, d_ff, rng),
        "ln_1": make_layer_norm_params(d_model),  # before MHA
        "ln_2": make_layer_norm_params(d_model),  # before FFN
    }


def encoder_block(X: np.ndarray, params: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """
    One encoder block, pre-norm variant.

    Layout (pre-norm):
        x + MultiHeadAttention(LayerNorm(x))
        followed by
        x + FFN(LayerNorm(x))

    Parameters
    ----------
    X : (T, d_model) input.
    params : dict from `make_encoder_block_params`.
    mask : optional (T, T) attention mask (for padding, typically).

    Returns
    -------
    (T, d_model) output.
    """
    # ---- Sublayer 1: multi-head self-attention ----
    # Pre-norm: normalize FIRST, then apply MHA. The unnormalized X is what
    # gets added back at the residual, preserving the gradient highway.
    a = layer_norm(X, params["ln_1"])
    a, _ = multi_head_attention(a, params["mha"], mask=mask)
    X = X + a  # residual

    # ---- Sublayer 2: position-wise feed-forward ----
    f = layer_norm(X, params["ln_2"])
    f = feed_forward(f, params["ffn"])
    X = X + f  # residual

    return X
```

This block is the unit cell of the transformer. Stacking N of them produces the full encoder.

## 10. Stacking Blocks: The Encoder

The full encoder is:

1. Token embedding: `(T,) integer IDs → (T, d_model) vectors`.
2. Add positional encoding.
3. (Optional) Dropout.
4. `N` encoder blocks in sequence.
5. (Optional) Final LayerNorm.

The original paper used `N=6` blocks for both encoder and decoder. Modern large models use anywhere from 12 (BERT-base, GPT-2 small) to 96+ (large-scale LLMs).

```python
# encoder.py
import numpy as np

from embeddings import make_embedding_matrix
from positional_encoding import sinusoidal_positional_encoding
from encoder_block import make_encoder_block_params, encoder_block
from layer_norm import make_layer_norm_params, layer_norm


def make_encoder_params(vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, rng) -> dict:
    """
    Allocate all parameters for an N-layer encoder.
    """
    return {
        "embedding": make_embedding_matrix(vocab_size, d_model),
        "blocks": [
            make_encoder_block_params(d_model, n_heads, d_ff, rng)
            for _ in range(n_layers)
        ],
        "final_ln": make_layer_norm_params(d_model),  # final pre-output LayerNorm
        "max_len": 1024,
    }


def encoder(token_ids: list[int], params: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Run a sequence of token IDs through the entire encoder stack.

    Parameters
    ----------
    token_ids : list of integer token IDs, length T.
    params : dict from `make_encoder_params`.
    mask : optional (T, T) attention mask.

    Returns
    -------
    (T, d_model) — final per-token contextualized representations.
    """
    # Step 1: look up token embeddings.
    X = params["embedding"][token_ids]  # (T, d_model)

    # Step 2: add positional encoding. We compute the PE matrix on the fly;
    # in production you'd cache it, but at our scale this is fine.
    T, d_model = X.shape
    PE = sinusoidal_positional_encoding(params["max_len"], d_model)
    X = X + PE[:T]

    # Step 3: pass through each encoder block in order. Each block preserves
    # shape, so we can chain them by simple reassignment.
    for block_params in params["blocks"]:
        X = encoder_block(X, block_params, mask=mask)

    # Step 4: final LayerNorm. This is standard with pre-norm: because
    # internal residuals are unnormalized, we add one final normalization
    # before the output is consumed.
    X = layer_norm(X, params["final_ln"])

    return X
```

That is a complete encoder, the same architecture as BERT (modulo training objective). For encoder-only tasks — text classification, named-entity recognition, sentence embeddings — this is the entire model. We just need a final task-specific head on top of `X` (e.g. a linear classifier on the `[CLS]` token's vector).

## 11. The Decoder and Cross-Attention (briefly)

The decoder has the same overall structure as the encoder but two changes:

1. **Masked self-attention.** The decoder generates output tokens left-to-right, one at a time. When predicting token `t`, it must not see tokens at positions `t+1, t+2, ...` — that would be cheating (the model would learn to copy the future). We enforce this with a **causal mask**: an upper-triangular `(T, T)` boolean matrix that suppresses attention from query position `i` to key position `j > i`.

   ```python
   # Causal mask: True where attention should be SUPPRESSED (j > i).
   causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
   ```

   Combined with our scaled-attention function from section 4.6, this gives masked self-attention.

2. **Cross-attention.** A second attention sublayer per block, where queries come from the decoder's own state but keys and values come from the encoder's output. This is how the decoder "looks at" the source sequence in a translation task: each output token can attend to any input token. Mechanically it is the same `scaled_dot_product_attention(Q, K, V)` we already wrote, but with `Q` from one source and `K, V` from another.

So the decoder block has *three* sublayers in sequence: masked self-attention, cross-attention, FFN — each with its own residual + LayerNorm.

```python
# decoder_block.py — sketch
def decoder_block(x, encoder_output, params, causal_mask):
    # 1. Masked self-attention: queries, keys, values all from x.
    a = layer_norm(x, params["ln_1"])
    a, _ = multi_head_attention(a, params["self_mha"], mask=causal_mask)
    x = x + a

    # 2. Cross-attention: queries from x, keys and values from encoder.
    # (We would need to extend multi_head_attention to accept separate Q and K,V sources;
    # the change is small — split X into X_q and X_kv before projecting.)
    c = layer_norm(x, params["ln_2"])
    c, _ = cross_attention(c, encoder_output, params["cross_mha"])
    x = x + c

    # 3. Feed-forward.
    f = layer_norm(x, params["ln_3"])
    f = feed_forward(f, params["ffn"])
    x = x + f

    return x
```

Encoder-only models (BERT) use only the encoder. Decoder-only models (GPT family) use only the decoder, dropping cross-attention since there is no encoder; the masked self-attention is the only attention. Encoder-decoder models (the original transformer, T5) use both. The architectural building blocks are the same; the choice depends on the task.

For simplicity, the rest of this tutorial works with a **decoder-only language model** — the GPT-style design — because language modeling on raw text is the cleanest training task to demonstrate end-to-end.

## 12. Putting It All Together: A Complete Transformer

We now switch from NumPy to PyTorch. NumPy was great for showing that everything is just matrix math, but to actually train we need autodiff and GPU support. The PyTorch version below is a faithful translation of everything we just built, plus the language-modeling head.

```python
# transformer.py
# A decoder-only (GPT-style) transformer in PyTorch.
# Every component corresponds line-for-line to the NumPy versions in earlier sections.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention, with optional causal masking.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # per-head dimensionality

        # Three projections for Q, K, V, stored as a single fused linear layer
        # for efficiency. nn.Linear includes a bias by default; we follow the
        # convention of having bias here (some implementations omit it).
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=True)

        # Output projection that mixes information across heads after concatenation.
        self.W_O = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x : (batch, T, d_model)
        mask : (T, T) boolean, True where attention should be SUPPRESSED.
               Will broadcast over batch and heads.
        Returns: (batch, T, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V in one shot, then split.
        # qkv has shape (B, T, 3*d_model); we split along the last axis.
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model)

        # Reshape to (B, n_heads, T, d_k) for parallel per-head computation.
        # The view + transpose pattern is the standard PyTorch idiom for this.
        def reshape_for_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        q = reshape_for_heads(q)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        # Scaled dot-product attention.
        # q @ k.transpose: (B, h, T, d_k) @ (B, h, d_k, T) = (B, h, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask: True positions in the mask become -inf in scores,
        # which after softmax gives weight 0.
        if mask is not None:
            # mask shape (T, T) broadcasts to (B, h, T, T).
            scores = scores.masked_fill(mask, float("-inf"))

        # Row-wise softmax: each query distributes 1 unit of attention.
        attn = F.softmax(scores, dim=-1)

        # Weighted average of values.
        # attn (B, h, T, T) @ v (B, h, T, d_k) = (B, h, T, d_k).
        out = attn @ v

        # Concatenate heads: (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model).
        # `.contiguous()` is required before .view() after a transpose because
        # transpose changes the stride pattern and view requires contiguous memory.
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Output projection.
        return self.W_O(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: Linear -> GELU -> Linear.
    GELU rather than ReLU is the modern default; both work.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU is a smoother variant of ReLU and is now standard in transformers.
        # Empirically it gives slightly better results; the difference is small
        # but free.
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    """
    One decoder-only transformer block: pre-norm masked self-attention, then
    pre-norm FFN, each with residual.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Sublayer 1: masked self-attention with residual.
        # Pre-norm: normalize, then attend, then add the original x back.
        x = x + self.attn(self.ln_1(x), mask=mask)
        # Sublayer 2: feed-forward with residual.
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerLM(nn.Module):
    """
    A decoder-only language model. Predicts the next token given the previous
    tokens. Architecture: token embedding + learned positional embedding,
    N decoder blocks, final LayerNorm, language-modeling head (which we tie to
    the input embedding to save parameters — a standard trick).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 256,
    ):
        super().__init__()
        self.max_len = max_len

        # Token embedding: V x d_model lookup table.
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding: max_len x d_model lookup table.
        # We use learned embeddings here rather than sinusoidal because (a)
        # they are simpler in PyTorch and (b) modern practice favors them
        # for short contexts. For very long contexts, RoPE or ALiBi are better,
        # but those are out of scope for this tutorial.
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Stack of decoder blocks.
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Final layer norm before the LM head.
        self.ln_final = nn.LayerNorm(d_model)

        # LM head: project from d_model back to vocab_size logits.
        # Weight tying: we share the matrix with the input embedding.
        # `lm_head` is computed as x @ tok_emb.weight.T inside `forward`,
        # so there is no separate parameter — a common trick that saves
        # vocab_size * d_model parameters and slightly improves performance.

        # Pre-build the causal mask once; we'll slice it per forward pass.
        # `register_buffer` puts the tensor on the right device with the model
        # but doesn't treat it as a parameter (no gradient).
        causal = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, T) long tensor of token IDs.
        Returns logits of shape (B, T, vocab_size).
        """
        B, T = token_ids.shape
        assert T <= self.max_len, "sequence longer than max_len"

        # Token embeddings: (B, T) -> (B, T, d_model).
        tok = self.tok_emb(token_ids)

        # Position indices: 0, 1, 2, ..., T-1 broadcast over batch.
        pos = torch.arange(T, device=token_ids.device)
        pos_e = self.pos_emb(pos)  # (T, d_model)

        # Sum: positional info added to token info, both in d_model space.
        x = tok + pos_e  # broadcasts over batch

        # Causal mask sliced to the actual sequence length.
        mask = self.causal_mask[:T, :T]

        # Pass through every decoder block.
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final norm.
        x = self.ln_final(x)

        # LM head with weight tying: x @ tok_emb.weight.T.
        # x is (B, T, d_model), tok_emb.weight is (V, d_model), so its transpose is
        # (d_model, V) and the product is (B, T, V) — the logits.
        logits = x @ self.tok_emb.weight.T

        return logits
```

That is a complete, runnable, decoder-only transformer language model. Every architectural piece from sections 2 through 11 is here. Total parameter count for the default config (d_model=128, 4 heads, 4 layers, vocab=128) is roughly 800k parameters — small enough to train on a CPU in minutes.

## 13. Training a Tiny Transformer on Real Text

To make this concrete, we train the model on a short paragraph of real English. The task is **next-character prediction**: given the previous characters, predict the next one. After training we sample new text from the model and inspect the attention patterns.

The text we train on: the opening of Lewis Carroll's *Alice's Adventures in Wonderland*. It is short (so training is fast), has rich English structure (so the model learns something interesting), and is in the public domain.

```python
# train_tiny.py
import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import CharTokenizer
from transformer import TransformerLM


# A short paragraph of real English. Repeat it a few times so we have enough
# data for the model to find regularities. In a real setting you would use
# millions or billions of characters; here we are just demonstrating the loop.
TEXT = (
    "Alice was beginning to get very tired of sitting by her sister on the "
    "bank, and of having nothing to do: once or twice she had peeped into the "
    "book her sister was reading, but it had no pictures or conversations in "
    "it, 'and what is the use of a book,' thought Alice 'without pictures or "
    "conversation?'"
) * 50  # repeat to give the model more training data


# ---- 1. Tokenize ----
tok = CharTokenizer(TEXT)
data = torch.tensor(tok.encode(TEXT), dtype=torch.long)
print(f"Vocab size: {tok.vocab_size}")
print(f"Total tokens: {data.numel()}")


# ---- 2. Hyperparameters ----
BLOCK_SIZE = 64       # context length: how many previous tokens the model sees
BATCH_SIZE = 32
N_STEPS    = 2000
LR         = 3e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) pairs from `data`.

    For each sample, we pick a random starting index `i`, take
    `data[i : i + block_size]` as the input and `data[i+1 : i+1+block_size]`
    as the target. The model's job is to predict, at every position, the
    NEXT character — that is, target[t] is the character that comes after input[t].
    """
    # Random starting indices, one per sample in the batch.
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    # Stack the input and target sequences.
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---- 3. Build the model ----
model = TransformerLM(
    vocab_size=tok.vocab_size,
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=4,
    max_len=BLOCK_SIZE,
).to(DEVICE)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")

# AdamW is the standard transformer optimizer: Adam with decoupled weight decay.
optimizer = optim.AdamW(model.parameters(), lr=LR)


# ---- 4. Training loop ----
model.train()
for step in range(N_STEPS):
    x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)

    # Forward pass: get logits at every position.
    logits = model(x)  # (B, T, V)

    # Cross-entropy loss between the logits and the next-token targets.
    # F.cross_entropy expects (N, V) and (N,) — so we flatten the batch and time dims.
    loss = nn.functional.cross_entropy(
        logits.view(-1, tok.vocab_size),  # (B*T, V)
        y.view(-1),                       # (B*T,)
    )

    # Standard backprop step.
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to prevent occasional spikes from destabilizing training.
    # 1.0 is a common default for transformers.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")


# ---- 5. Sampling ----
@torch.no_grad()
def sample(model: TransformerLM, prompt: str, n_new_tokens: int = 200, temperature: float = 1.0) -> str:
    """
    Autoregressive sampling: generate `n_new_tokens` characters following `prompt`.

    At each step we feed the current sequence into the model, take the logits
    for the LAST position, sample a token from them, and append it to the sequence.
    """
    model.eval()
    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)

    for _ in range(n_new_tokens):
        # If the sequence is longer than the context, truncate from the left.
        x_cond = x[:, -BLOCK_SIZE:]
        logits = model(x_cond)               # (1, T, V)
        logits = logits[:, -1, :] / temperature  # (1, V) — only the last step matters
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        x = torch.cat([x, next_id], dim=1)

    return tok.decode(x[0].tolist())


# Generate a sample to see what the model has learned.
print("\n--- Sample after training ---")
print(sample(model, prompt="Alice was ", n_new_tokens=300))
```

After 2000 steps on this tiny dataset, the model produces output that is recognizably English-shaped: real words appear, basic punctuation rhythms emerge, and you may even see fragments of the original text reproduced (the dataset is small enough that the model partly memorizes — that is expected and fine for a demo). On a CPU this run takes a few minutes; on a GPU, seconds.

To inspect the trained attention patterns, hook into one of the blocks and capture the attention weights from a forward pass:

```python
# inspect_attention.py
import torch
import matplotlib.pyplot as plt
import numpy as np

# After training the model from train_tiny.py, capture attention weights.
# We modify the MultiHeadAttention.forward to optionally return attention,
# OR we monkey-patch with a forward hook. Below is the hook approach,
# which avoids modifying the original module.

attention_storage = {}

def make_hook(name):
    def hook(module, inputs, output):
        # We need the raw attention probabilities. Easiest path is to
        # recompute them from the inputs to the module — but in a hook we
        # only see the output, not internals. So we compute them externally
        # by inspecting the module's projections directly.
        pass  # placeholder; in practice modify MHA to return attn
    return hook


# A more practical approach: modify MultiHeadAttention.forward to also return
# the attention tensor when a flag is passed. Then:
def collect_attention(model, prompt: str, layer_idx: int, head_idx: int) -> tuple[np.ndarray, list[str]]:
    """
    Run a forward pass and return the (T, T) attention matrix for the given
    layer and head, plus the token labels.
    """
    from tokenizer import CharTokenizer  # adjust import as needed in your project
    # Implementation requires extending MultiHeadAttention to return attn;
    # see comments in transformer.py for the necessary one-line change.
    raise NotImplementedError(
        "Extend MultiHeadAttention.forward to also return the attention tensor "
        "before adding visualization. The change is one line: replace "
        "`return self.W_O(out)` with `return self.W_O(out), attn` and update "
        "callers to unpack the tuple."
    )
```

The simplest way to enable visualization is to extend `MultiHeadAttention.forward` to also return `attn` (the post-softmax attention tensor of shape `(B, h, T, T)`). Then for any input sequence, plot one head's attention matrix using the `plot_attention` function from section 4.7. In a trained model on natural text, you will see patterns like:

- **Diagonal attention** (each token attends to itself or the immediately previous token) — a "previous token" head, often present.
- **Vertical stripes** — many tokens attending to one important token like the start-of-sequence or a punctuation mark.
- **Off-diagonal spots** — attention to specific earlier tokens, e.g. closing quotes attending to the matching opening quote.

These patterns are not pre-programmed; they emerge from training. That emergence is the heart of why transformers work.

## 14. Where to Go Next

This tutorial covered the architecture in *Attention Is All You Need* (2017) and the small modernizations that have become standard. Real-world systems extend it in several directions, none of which change the core ideas:

The **scaling axis** — bigger versions of exactly this architecture. GPT-3 is `d_model=12288`, 96 layers, 96 heads. The architecture is unchanged. What scales: data, parameters, compute, training time. The behavior that emerges at scale (in-context learning, instruction following, chain-of-thought) is qualitatively different but mechanistically just "the same architecture, larger".

The **efficiency axis** — making attention cheaper. The naive `softmax(QK^T/√d_k)V` is O(T²) in time and memory, which becomes prohibitive for long sequences. Improvements include FlashAttention (an IO-optimized kernel that reorders the same computation to avoid materializing the full T×T attention matrix), grouped-query attention (sharing K and V across multiple Q heads, smaller cache for inference), and sparse-attention variants (Longformer, BigBird) that restrict each query to a subset of keys.

The **positional axis** — better ways to inject position. Sinusoidal and learned absolute embeddings (which we used) work but generalize poorly to sequences longer than seen at training. RoPE (Rotary Position Embeddings) and ALiBi (Attention with Linear Biases) encode *relative* position by modifying attention itself rather than adding to embeddings, and they extrapolate much better.

The **architectural axis** — variants beyond decoder-only. Encoder-decoder models like T5 are still strong on translation and structured generation. Mixture-of-Experts (MoE) replaces the dense FFN with a router selecting from many "expert" FFNs, growing capacity without growing per-token compute. State-space models like Mamba revive the recurrence idea with hardware-friendly designs and compete with transformers on some benchmarks.

For deeper study: the Karpathy nanoGPT codebase is the cleanest minimal-but-real implementation of a GPT-style transformer; the *Attention Is All You Need* paper repays direct reading once you have built the thing yourself; the Anthropic mechanistic-interpretability papers ("A Mathematical Framework for Transformer Circuits", "In-context Learning and Induction Heads") show what trained attention heads actually compute and are deeply rewarding once the architecture is internalized.

But the architecture in this tutorial is not a stepping stone to something more complicated — it is the thing itself. Every model named in the previous paragraph is, structurally, what we built. Understand this, and you understand the rest.
