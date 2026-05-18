---
status: active
type: log
description: Append-only working log for the philnlp_course repository — chronological session entries covering significant changes, decisions, and unit-doc updates.
label: [planning]
injection: informational
volatility: evolving
scope: general
last_checked: '2026-05-18'
---
# 🎓 Philosophy of NLP — Working Log

Append-only working history. Newest entries first. Add an entry whenever a difficult problem is solved, a significant change is made, or a major task is completed.

---

## 2026-05-18 — Built Unit 3 contextual embeddings notebook (Skip-Gram + GloVe), made it Colab-ready

- **Task:** The starter `03_contextual_embeddings/Skip-Gram.ipynb` had the skip-gram equations and a 9-word toy demo but no real training and no semantic analogy demo. The unit needed an end-to-end pedagogical arc that actually lands `king − man + woman ≈ queen` on real data, plus a Colab path for students without a local Python environment.
- **Outcome:**
  - **Rebuilt: [03_contextual_embeddings/Skip-Gram.ipynb](03_contextual_embeddings/Skip-Gram.ipynb)** — 19 cells covering distributional hypothesis → skip-gram architecture and cross-entropy loss equations → pure-NumPy toy → motivation for subsampling + negative sampling → SGNS math → load NLTK Brown (~1M tokens) → numpy mini-batched SGNS training (~100s on laptop CPU) → nearest-neighbour + PCA inspection → honest reckoning that 1M tokens isn't enough → load pretrained GloVe (gensim, `glove-wiki-gigaword-100`, ~134 MB) → classic analogies (king–man+woman → queen ✅; paris–france+germany → berlin ✅; tokyo–japan+italy → rome ✅; etc.) → bridge to transformers via the bank/bank polysemy problem.
  - **Colab readiness:** added a Setup cell that detects Colab via `'google.colab' in sys.modules`, pip-installs `gensim` (Colab dropped it from the base image around 2023), centralises the macOS SSL workaround, and defines `SMOKE_TEST` + `AUTO_DISCONNECT` flags. `SMOKE_TEST=True` caps Brown at 100k tokens and runs 1 epoch — full notebook in <30s for iteration. Final cell optionally disconnects the runtime on paid Colab plans.
  - **Updated: [03_contextual_embeddings/CONTEXTUAL_EMBEDDINGS.md](03_contextual_embeddings/CONTEXTUAL_EMBEDDINGS.md)** — status bumped from ⬜ Planned to 🟡 In progress with a notebook content summary; ELMo/BERT contextual section explicitly still pending.
  - **Updated: [README.md](README.md)** — Unit 3 status row matches the new state.
  - **Updated: [requirements.txt](requirements.txt)** — added `scikit-learn>=1.4` (PCA) and `gensim>=4.3` (GloVe loader) under a new Unit 3 section.
  - **Memory (auto-memory):** added `feedback_avoid_local_compute.md` capturing the rule "don't re-execute compute-heavy notebooks locally in this repo; user runs them on Google Colab — verify with smoke-test scripts instead."
- **Key decisions:**
  - **Pure NumPy implementation, no PyTorch.** Local Anaconda's torch was broken (`libtorch_cpu.dylib` missing) and the project `.venv` listed in CLAUDE.md doesn't exist locally. NumPy SGNS with mini-batched `np.einsum` + `np.add.at` trains the full Brown corpus in ~100s on CPU — fast enough for the lesson and more transparent pedagogically than autograd. Pure-NumPy also means the notebook stays single-language with the rest of Unit 3's planned content.
  - **Skip-Gram with Negative Sampling rather than full softmax.** The original toy used full softmax and was advertised as the "real" implementation in the markdown cells. For 8k-vocab Brown that would be intractable; SGNS is what Word2Vec actually uses (Mikolov 2013b) and what the linked PDFs in the folder describe.
  - **Pretrained GloVe via `gensim.downloader.load("glove-wiki-gigaword-100")`** rather than training analogy-capable vectors from scratch. Honest framing: 1M tokens of Brown gives the model only ~88 sightings of `king`, far too few for the analogy structure. The notebook is explicit that the wow demo uses 6B-token pretrained vectors, not the from-scratch Brown training — preserving the "build it yourself" pedagogy while still landing the classic analogy.
  - **NLTK Brown over Gutenberg/text8.** Brown is licensed, modest size, and is what NLTK's `nltk.download('brown')` gives you out of the box. Gutenberg's archaic vocabulary would have muddled the king/queen sanity-check; text8 would have added a 30 MB download step.
  - **Single combined notebook (not split per concept)** — overrode the saved "default to .py scripts" feedback from earlier sessions because the user explicitly said "notebook" for this unit; logged the override in the AskUserQuestion exchange.
- **Follow-up:**
  - Verify the Colab path end-to-end on Colab itself — the Setup cell's `pip install gensim` branch and NLTK SSL handling are tested locally but not on Colab's actual filesystem/network. First run on Colab will refresh the cached outputs (currently show local-laptop training timings of ~33s/epoch; Colab CPU likely closer to 50-60s/epoch).
  - The contextual-embedding section of `CONTEXTUAL_EMBEDDINGS.md` (ELMo / BERT-style token-dependent vectors, visualisation tooling) is still planned — would be a natural standalone notebook in the same folder.
  - **KB write candidate (deferred — kb_mcp not connected in this session):** `NOTEBOOK_WRITING_SKILL.md` could be updated to enumerate "packages that need an in-notebook pip install on current Colab" (the gensim case is the load-bearing example). Marginal; defer until a second instance shows up.

---

## Notes on this session's workflow execution

- **Phase 5 (Knowledge Capture):** kb_mcp MCP server is not connected in this environment, so no KB imports or updates were performed. The notebook-writing skill update candidate above is the only deferred KB write.
- **Phase 7 (KB Performance Feedback):** kb_mcp not connected; `knowledge_base_record_performance` calls were not possible. Notable for the user when they next run a session with kb_mcp available: the gensim/Colab-base-image fact surfaced during this session and is the kind of constraint a future agent shipping a Colab notebook with gensim might rediscover.
