"""
🔤 N-Gram Language Models: A Hands-On Tutorial
================================================

The practical companion to NGRAMS.md. We build n-gram language models from
scratch on real text — Jane Austen's *Emma*, via NLTK's Gutenberg corpus —
and then compare our hand-rolled implementation to NLTK's `lm` package.

Run it end-to-end:

    python 01_ngrams_tutorial.py

It is intended to be read while running. Each numbered section is a step in the
pedagogy: tokenize → count → estimate → smooth → generate → evaluate. Print
statements after every meaningful computation show shapes, intermediate values,
and sanity checks.

The big question
----------------
A language model assigns a probability to a sequence of words. With nothing but
counts, can we get something that (a) distinguishes likely from unlikely
sentences, (b) generates plausible new text, (c) tells us anything about
meaning?

The recipe is: count word sequences, divide. That is it. Everything else
(smoothing, back-off, sampling, evaluation) is damage control for the fact that
you will never see all possible sequences in a finite corpus.

Philosophical hook
------------------
N-grams are the purest expression of the distributional hypothesis — meaning is
co-occurrence. No semantics, no syntax, no world knowledge, just frequency. The
fact that this gets us anywhere at all is the surprising philosophical
observation that motivates the rest of the course.
"""

# =====================================================================
# 0. Setup
# =====================================================================
#
# We need NLTK for the corpus + tokenizers, and matplotlib for one frequency
# plot. The first run will also download three small NLTK data packages
# (`gutenberg`, `punkt`, `punkt_tab`) — under 5 MB combined, cached to
# ~/nltk_data, no-op on subsequent runs.

import random
import sys
import subprocess
from collections import Counter, defaultdict


def _ensure_packages():
    # Self-bootstrap: if nltk or matplotlib aren't importable, install them into
    # *this* interpreter (not whatever pip happens to be first on PATH).
    # Done as a function so the import errors are caught cleanly per package.
    for pkg in ("nltk", "matplotlib"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure_packages()

import nltk  # noqa: E402  — must come after install bootstrap
import matplotlib.pyplot as plt  # noqa: E402

# Pull down the corpus we'll train on and the tokenizer models.
# `quiet=True` suppresses the per-package download chatter.
# `punkt_tab` is needed by NLTK >= 3.9; downloading is a no-op on older versions.
for resource in ("gutenberg", "punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)

# Deterministic randomness — every sample we draw should be reproducible across
# runs. When you read a generated sentence below, you should see *the same one*
# when you re-run this file.
random.seed(42)

print("=" * 70)
print("Setup complete.")
print(f"NLTK version: {nltk.__version__}")
print("=" * 70)


# =====================================================================
# 1. Pick a corpus
# =====================================================================
#
# NLTK's `gutenberg` collection ships 18 public-domain books. We use Jane
# Austen's *Emma* — small enough to train on in seconds, large enough that
# bigram and trigram statistics are non-trivial. To feel how dramatically the
# LM changes with genre, swap in `melville-moby_dick.txt` or
# `shakespeare-hamlet.txt` below.

from nltk.corpus import gutenberg  # noqa: E402

print("\n[1] Loading corpus")
print("-" * 70)
print("Available Gutenberg texts:")
for fileid in gutenberg.fileids():
    print(f"  {fileid}")

# `raw()` returns one big string. We could use `gutenberg.words()` to get the
# pre-tokenized version, but the whole point of the next section is to make
# the preprocessing visible — so we do it ourselves.
text = gutenberg.raw("austen-emma.txt")

print(
    f"\nLoaded austen-emma.txt: {len(text):,} characters, "
    f"{len(text.split()):,} whitespace-separated 'words' (before real tokenization)."
)
print("First 300 chars:")
print(repr(text[:300]))


# =====================================================================
# 2. Tokenize
# =====================================================================
#
# A language model is a distribution over sequences of *tokens*, so the first
# decision is "what counts as a token?" We use NLTK's word tokenizer, which
# handles punctuation sensibly (splits "don't" into "do" + "n't", keeps "Mr."
# together, treats "," and "." as their own tokens). Lowercasing folds "The"
# and "the" into one type.
#
# Crucially, we tokenize *sentence by sentence*. N-gram probabilities are
# estimated within sentences — we don't want the last word of one chapter to
# act as a predictor of the first word of the next.

from nltk.tokenize import sent_tokenize, word_tokenize  # noqa: E402

print("\n[2] Tokenizing")
print("-" * 70)

# `sent_tokenize` uses the trained 'punkt' model — it knows that "Mr." is not
# a sentence terminator, that a period inside a quote may or may not be, etc.
sentences = sent_tokenize(text)
print(f"Sentences detected: {len(sentences):,}")
print(f"First sentence: {sentences[0]!r}")

# Result: list of lists. Outer list = sentences. Inner list = lowercased tokens.
# Each inner list is one training example for the language model.
tokenized = [
    [tok.lower() for tok in word_tokenize(sent)]
    for sent in sentences
]

total_tokens = sum(len(s) for s in tokenized)
print(f"\nTotal tokens: {total_tokens:,}")
print(f"Average sentence length: {total_tokens / len(tokenized):.1f} tokens")
print("First tokenized sentence:")
print(tokenized[0])


# =====================================================================
# 3. Train / test split
# =====================================================================
#
# We hold out 10% of sentences for evaluation. The model never sees these
# during training, so perplexity later tells us how the model *generalises*,
# not how well it memorised.

print("\n[3] Train/test split")
print("-" * 70)

# Shuffle a copy — preserves the original `tokenized` list if we want it later.
# random.seed(42) above makes this split (and every later sample) reproducible.
shuffled = tokenized[:]
random.shuffle(shuffled)

split_idx = int(0.9 * len(shuffled))
train_sents = shuffled[:split_idx]
test_sents = shuffled[split_idx:]

print(f"Train sentences: {len(train_sents):,}")
print(f"Test sentences:  {len(test_sents):,}")


# =====================================================================
# 4. Unigrams — the simplest possible model
# =====================================================================
#
# A unigram model treats every word as independent of every other:
#
#     P(w_1, ..., w_n) = P(w_1) * P(w_2) * ... * P(w_n).
#
# Estimating it is just counting. This will be a *terrible* language model —
# it has no idea about word order, samples will be word salad — but it is a
# useful baseline.

print("\n[4] Unigram model")
print("-" * 70)

# For unigrams we don't care about sentence boundaries — flatten everything
# into one big list of tokens.
train_tokens = [tok for sent in train_sents for tok in sent]

# `Counter` is a specialised dict mapping each unique element to its frequency.
# This single line is the entire training step for a unigram model.
unigram_counts = Counter(train_tokens)
total_unigrams = sum(unigram_counts.values())  # == len(train_tokens)

print(f"Total tokens in training set: {total_unigrams:,}")
print(f"Unique tokens (vocabulary size V): {len(unigram_counts):,}")
print("\nTop 15 most frequent tokens:")
for word, count in unigram_counts.most_common(15):
    print(f"  {word!r:>15}  {count:>6,}  ({count/total_unigrams:.2%})")

# Zipf's law in action: word frequencies follow a power-law distribution.
# A handful of tokens ('the', ',', '.', 'to', 'and', …) eat a huge fraction
# of all probability mass. Almost every other word is rare. This is *why*
# smoothing matters in later sections: in a finite corpus, most legitimate
# bigrams and trigrams will simply not appear.
top_n = 30
top_words = unigram_counts.most_common(top_n)
words, counts = zip(*top_words)

plt.figure(figsize=(10, 4))
plt.bar(range(top_n), counts)
plt.xticks(range(top_n), words, rotation=70)
plt.ylabel("count")
plt.title(f"Top {top_n} tokens in Emma (training split)")
plt.tight_layout()
plt.savefig("01_ngrams/unigram_zipf.png", dpi=100)
plt.close()
print("\nSaved frequency plot to 01_ngrams/unigram_zipf.png")

# --- Maximum-likelihood unigram probability ------------------------------
#
#     P_MLE(w) = count(w) / total_tokens.
#
# Turn the counter into a probability distribution by dividing each count by
# the total.
unigram_prob = {w: c / total_unigrams for w, c in unigram_counts.items()}

# Spot-check: probabilities should sum to 1.0 (modulo floating-point noise).
print(f"\nSum of unigram probabilities: {sum(unigram_prob.values()):.6f}")
print(f"P('the')        = {unigram_prob['the']:.5f}")
print(f"P('emma')       = {unigram_prob.get('emma', 0.0):.5f}")
print(
    f"P('philosophy') = {unigram_prob.get('philosophy', 0.0):.5f}  "
    f"← 0 if Austen never used the word in Emma."
)

# --- Sampling a unigram sentence ----------------------------------------
# To see what the model has learned, we draw words from the distribution.
# Because unigrams are word-independent, the output is word salad — but
# that's the point. It establishes a baseline of *how bad it sounds when
# context is ignored*.

def sample_unigram(n_words: int) -> str:
    # `random.choices` does weighted sampling — each word is picked with
    # probability proportional to its weight. Using raw counts as weights is
    # equivalent to sampling from the MLE distribution.
    sample_words = list(unigram_counts.keys())
    weights = list(unigram_counts.values())
    sampled = random.choices(sample_words, weights=weights, k=n_words)
    return " ".join(sampled)


print("\nUnigram samples (word salad — by design):")
for i in range(3):
    print(f"  Sample {i+1}: {sample_unigram(20)}")


# =====================================================================
# 5. Bigrams — adding one word of context
# =====================================================================
#
# A bigram model conditions each word on the immediately preceding one:
#
#     P(w_2 | w_1) = count(w_1, w_2) / count(w_1).
#
# To handle the start and end of sentences we pad each one with `<s>` (start)
# and `</s>` (end) symbols. Without that padding, the model wouldn't be able
# to learn which words tend to *begin* a sentence vs. occur mid-sentence.

from nltk.util import ngrams, pad_sequence  # noqa: E402

print("\n[5] Bigram model")
print("-" * 70)


def pad(sentence):
    # Order-2 padding adds one start and one end symbol (for bigrams). For
    # higher-order models we'd pad with more.
    return list(pad_sequence(
        sentence, n=2,
        pad_left=True, left_pad_symbol="<s>",
        pad_right=True, right_pad_symbol="</s>",
    ))


# bigram_counts[prev_word][next_word] = count of (prev_word, next_word).
# This nested-Counter shape makes conditional probability trivial — just
# normalize each prev_word's row by `context_counts[prev_word]`.
bigram_counts: dict[str, Counter] = defaultdict(Counter)
context_counts: Counter = Counter()  # count(prev_word) — MLE denominator

for sent in train_sents:
    padded = pad(sent)
    for w1, w2 in ngrams(padded, 2):
        bigram_counts[w1][w2] += 1
        context_counts[w1] += 1

# Sanity peek: what tends to follow 'she' in Emma?
print("Top 10 words following 'she':")
for word, count in bigram_counts["she"].most_common(10):
    p = count / context_counts["she"]
    print(f"  she {word!r:<10}  count={count:>4}  P={p:.3f}")


# --- Bigram MLE probability ----------------------------------------------

def bigram_prob_mle(w1: str, w2: str) -> float:
    # P(w2 | w1) = count(w1, w2) / count(w1).
    # If w1 was never seen, denominator is 0 — the probability is undefined.
    # If (w1, w2) was never seen, numerator is 0, and the *whole sentence*
    # containing that bigram gets probability 0. This is the zero-probability
    # problem that motivates smoothing in the next section.
    if context_counts[w1] == 0:
        return 0.0
    return bigram_counts[w1][w2] / context_counts[w1]


print(f"\nP('know' | 'i')          = {bigram_prob_mle('i', 'know'):.4f}")
print(f"P('was' | 'she')         = {bigram_prob_mle('she', 'was'):.4f}")
print(
    f"P('philosopher' | 'the') = {bigram_prob_mle('the', 'philosopher'):.4f}  "
    f"← almost certainly 0 — 'the philosopher' never appears in Emma."
)


# =====================================================================
# 6. The zero-probability problem & Laplace smoothing
# =====================================================================
#
# A single unseen bigram makes the probability of an entire sentence zero.
# That's clearly wrong: "the philosopher" is unusual in Emma but not
# *impossible*.
#
# Laplace (add-1) smoothing pretends we saw every bigram one extra time:
#
#     P_Laplace(w_2 | w_1) = (count(w_1, w_2) + 1) / (count(w_1) + V),
#
# where V is the vocabulary size. The +V in the denominator keeps the row
# summing to 1.
#
# Laplace is the simplest possible smoother. It over-corrects (it gives too
# much probability mass to unseen bigrams), but it captures the qualitative
# behaviour smoothing is supposed to give us and is a fine starting point.

print("\n[6] Laplace smoothing")
print("-" * 70)

V = len(unigram_counts)  # vocabulary size


def bigram_prob_laplace(w1: str, w2: str) -> float:
    # Add-1: every bigram is treated as if it had occurred at least once.
    return (bigram_counts[w1][w2] + 1) / (context_counts[w1] + V)


print(
    f"P_Laplace('philosopher' | 'the') = {bigram_prob_laplace('the', 'philosopher'):.6f}  "
    f"← no longer zero, but still tiny."
)
print(f"P_MLE    ('know'         | 'i')  = {bigram_prob_mle('i', 'know'):.4f}")
print(
    f"P_Laplace('know'         | 'i')  = {bigram_prob_laplace('i', 'know'):.4f}  "
    f"← shrunk slightly: mass was redistributed to unseen bigrams."
)


# =====================================================================
# 7. Using NLTK's `lm` module
# =====================================================================
#
# We built a bigram model by hand because seeing the dictionaries is how the
# math becomes concrete. In practice you'd reach for NLTK's `lm` package: it
# handles padding, vocabulary, multiple smoothing schemes, perplexity, and
# generation behind one consistent API.
#
# We train three *trigram* models — MLE, Laplace, KneserNeyInterpolated — on
# the same data and compare them. Kneser–Ney is the standard high-quality
# n-gram smoother; it's what production n-gram systems used before neural
# language models took over.

from nltk.lm.preprocessing import padded_everygram_pipeline  # noqa: E402
from nltk.lm import MLE, Laplace, KneserNeyInterpolated  # noqa: E402

print("\n[7] NLTK lm — three trigram models")
print("-" * 70)

N = 3  # trigram order — P(w_3 | w_1, w_2)


def make_data():
    # padded_everygram_pipeline returns two iterators:
    #   - the n-grams to fit on ("everygrams" = 1-grams + 2-grams + ... + n-grams)
    #   - the padded vocabulary stream (sentences padded with <s> and </s>).
    # Iterators are consumed on .fit(), so we regenerate per model.
    return padded_everygram_pipeline(N, train_sents)


train_ngrams, padded_vocab = make_data()
mle_model = MLE(N)
mle_model.fit(train_ngrams, padded_vocab)

train_ngrams, padded_vocab = make_data()
laplace_model = Laplace(N)
laplace_model.fit(train_ngrams, padded_vocab)

train_ngrams, padded_vocab = make_data()
kn_model = KneserNeyInterpolated(N)
kn_model.fit(train_ngrams, padded_vocab)

print(f"Vocab size (incl. <s>, </s>): {len(mle_model.vocab):,}\n")
print(f"  MLE       P('know' | 'i')          = {mle_model.score('know', ['i']):.4f}")
print(f"  Laplace   P('know' | 'i')          = {laplace_model.score('know', ['i']):.4f}")
print(f"  KneserNey P('know' | 'i')          = {kn_model.score('know', ['i']):.4f}")
print()
print(f"  MLE       P('philosopher' | 'the') = {mle_model.score('philosopher', ['the']):.4f}  ← zero")
print(f"  Laplace   P('philosopher' | 'the') = {laplace_model.score('philosopher', ['the']):.6f}")
print(f"  KneserNey P('philosopher' | 'the') = {kn_model.score('philosopher', ['the']):.6f}")


# =====================================================================
# 8. Generating sentences
# =====================================================================
#
# Each model has a `.generate()` method that samples a sequence one token at
# a time. The trigram conditions on the previous *two* tokens, so the output
# is markedly more coherent than the unigram word salad earlier — though still
# very local. The model has no notion of *what the sentence is about*; it just
# knows what tends to follow what.

print("\n[8] Sentence generation")
print("-" * 70)


def generate_sentence(model, max_len: int = 25, seed: int | None = None) -> str:
    """Generate a sentence as a single string, stopping at </s> or max_len tokens."""
    # `random_seed` makes the output reproducible — pass different seeds for
    # different generations from the same model.
    tokens = model.generate(max_len, random_seed=seed)
    out: list[str] = []
    for tok in tokens:
        if tok == "</s>":
            break
        out.append(tok)
    return " ".join(out)


print("MLE trigram:")
for s in range(3):
    print(f"  • {generate_sentence(mle_model, seed=s)}")

print("\nLaplace trigram:")
for s in range(3):
    print(f"  • {generate_sentence(laplace_model, seed=s)}")

# NOTE: We skip generation from `kn_model` here. `KneserNeyInterpolated.generate()`
# is *extremely* slow in NLTK because it backs off recursively through all orders
# and has to score the entire vocabulary at every step — on a 7,600-word vocab a
# single 25-token sentence can take 30+ seconds. The qualitative output is similar
# to the Laplace trigram anyway. We still evaluate KN's *perplexity* in the next
# section, which is the more informative comparison.
print("\nKneser–Ney trigram: (sampling skipped — see comment in source; KN .generate() is impractically slow on this vocab size)")


# =====================================================================
# 9. Evaluating with perplexity
# =====================================================================
#
# Perplexity is the standard intrinsic evaluation for a language model.
# Informally: it's the average number of equally-likely next words the model
# thinks it's choosing between. *Lower is better.*
#
#     PP(W) = P(w_1, ..., w_N) ** (-1/N).
#
# A model that always assigns probability 1 to the correct next word has
# perplexity 1. A model that spreads probability uniformly across V words has
# perplexity V.
#
# We evaluate on test_sents — sentences the models have never seen. MLE will
# explode (perplexity = ∞) on any test sentence containing an unseen trigram.
# Smoothed models will not. This is the practical payoff of smoothing.

from nltk.lm.preprocessing import padded_everygrams  # noqa: E402

print("\n[9] Perplexity")
print("-" * 70)


# NLTK's KneserNey.perplexity is *slow* — it backs off recursively per n-gram.
# On the full ~785-sentence test set it takes several minutes. For pedagogical
# purposes we subsample. Set N_EVAL = len(test_sents) below to run on everything.
N_EVAL = 50
eval_sents = test_sents[:N_EVAL]


def avg_perplexity(model, sents):
    """Mean per-sentence perplexity; also reports how many blew up to infinity."""
    scores = []
    for sent in sents:
        # padded_everygrams yields the n-grams of this sentence, padded.
        # We materialize the iterator with list(...) so model.perplexity can consume it.
        ngs = list(padded_everygrams(N, sent))
        pp = model.perplexity(ngs)
        scores.append(pp)
    finite = [p for p in scores if p != float("inf")]
    return {
        "mean_finite": sum(finite) / len(finite) if finite else float("nan"),
        "n_infinite": len(scores) - len(finite),
        "n_total": len(scores),
    }


print(f"Perplexity on {N_EVAL} held-out test sentences (lower is better):")
for name, model in [("MLE", mle_model), ("Laplace", laplace_model), ("KneserNey", kn_model)]:
    print(f"  evaluating {name}…", flush=True)
    stats = avg_perplexity(model, eval_sents)
    print(
        f"  {name:>10}: mean (finite) = {stats['mean_finite']:>10.2f}  "
        f"(∞ on {stats['n_infinite']}/{stats['n_total']} sentences)"
    )


# =====================================================================
# 10. What n-grams can and can't do
# =====================================================================
#
# What we got, almost for free:
#   • A working language model from nothing but counts and division.
#   • Generated text that is *locally* fluent — adjacent words make sense.
#   • A quantitative evaluation (perplexity) that distinguishes good and bad
#     models, and that punishes models which assign probability 0 to plausible
#     events.
#
# What's missing:
#   • Long-range dependencies. A trigram knows the previous two tokens. It
#     cannot remember that a sentence started with "Mr. Knightley" twenty
#     words ago and use that to pick the right pronoun later.
#   • Generalisation across similar words. "the philosopher" and "the scholar"
#     are completely unrelated events to a bigram model, even though they
#     pattern alike semantically. Every word is its own atomic symbol.
#   • Compositionality. N-grams have no notion that "not happy" and "unhappy"
#     mean roughly the same thing — or that "happy" and "happiness" share
#     anything at all.
#
# Bridge to the next unit.
# Hidden Markov models address the first limitation: they posit a hidden state
# that *summarises* arbitrary amounts of past context into a single latent
# variable. The next unit, ../02_hmm/HMM.md, builds them from scratch.
#
# A philosophical aside.
# Notice that we trained a model of *Emma* — a model that produces text
# resembling Jane Austen. Is that "knowing English", or is it "imitating one
# English author"? The same algorithm trained on legal contracts, Twitter, or
# Hegel would produce three radically different models. There is no neutral
# corpus to learn from; every corpus encodes a perspective on what language
# *is*. That observation returns in every unit that follows.
#
# Try it yourself:
#   1. Swap 'austen-emma.txt' for 'melville-moby_dick.txt',
#      'shakespeare-hamlet.txt', or 'bible-kjv.txt' in section 1. How does the
#      generated text change? Where does each genre's vocabulary and syntax
#      break the model?
#   2. Drop the trigram order from 3 to 2 (set N = 2) and re-run. Does
#      perplexity get better or worse? Are samples more or less coherent?
#   3. Push the order up to 5. Generation will sound spookily Austen-like —
#      but check perplexity on test. What's going on?

print("\n" + "=" * 70)
print("Tutorial complete.")
print("=" * 70)
