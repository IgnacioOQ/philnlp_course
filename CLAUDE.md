# Transformer Tutorial — Claude Context

## Environment

- **Python:** Python 3.13.7. The venv lives **in the repo root**: `philnlp_course/.venv` (built from the python.org framework Python at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`). Interpreter: `./.venv/bin/python`.
- **Packages:** numpy, matplotlib, notebook, jupyterlab, ipykernel (transformer notebooks only use numpy + matplotlib). `torch` is **not** installed yet — it is only needed for the training sections (12–13); add it then.
- **Registered kernel:** `philnlp-transformers`, display name `.venv (philnlp 3.13.7)`, pointing at `./.venv/bin/python`. Re-create it with `./.venv/bin/python -m ipykernel install --user --name philnlp-transformers --display-name ".venv (philnlp 3.13.7)"`. New notebooks' `metadata.kernelspec` must use `{"display_name": ".venv (philnlp 3.13.7)", "language": "python", "name": "philnlp-transformers"}` — copy that block verbatim.
- **To execute a notebook headless:** `./.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=philnlp-transformers <nb>.ipynb`.
- **Do NOT use** the system Anaconda Python (`/Users/ignacio/anaconda3`, also the default `python3` on PATH) — it has a broken PyTorch install that crashes the Jupyter kernel. There is also an **old duplicate** of this tutorial at `/Users/ignacio/Desktop/Personal Apps/Transformer Tutorial/` (its own `.venv` and the now-stale `transformer-tutorial` kernel) — that is the pre-repo original; ignore it, this repo copy is canonical.

## Notebook conventions

- The transformer-tutorial notebooks live in [`04_transformers/`](04_transformers/).
- One `.ipynb` per tutorial section, named with a two-digit prefix: `04_transformers/01_tokenizer.ipynb`, `04_transformers/02_embeddings.ipynb`, `04_transformers/03_attention.ipynb`, etc.
- **When creating a new notebook**, always copy the `metadata.kernelspec` from an existing notebook. The correct value is:
  ```json
  {"display_name": ".venv (3.13.7)", "language": "python", "name": "python3"}
  ```
  Using the generic `"Python 3"` default causes VS Code to connect to the wrong kernel and loop in "Restarting Kernel".
- Every code cell must have:
  - Inline comments explaining **why** the code does what it does (not just what).
  - Print statements after every meaningful computation, with labels and sanity checks.

## Progress

| Notebook | Section | Status |
|---|---|---|
| `01_tokenizer.ipynb` | 2 — Tokenization | ✅ Done |
| `02_embeddings.ipynb` | 3 — Embeddings | ✅ Done |
| `03_attention.ipynb` | 4 — Attention (4.1–4.7) | ✅ Done |
| `04_multihead_attention.ipynb` | 5 — Multi-Head Attention | ✅ Done |
| `05_positional_encoding.ipynb` | 6 — Positional Encodings | ⬜ Next |
| `06_feedforward.ipynb` | 7 — Feed-Forward + Layer Norm | ⬜ |
| `07_encoder.ipynb` | 9–10 — Encoder Block + Stack | ⬜ |
| `08_transformer.ipynb` | 12–13 — Full PyTorch Transformer + Training | ⬜ |

## Shared utilities

- `tokenizer.py` — `CharTokenizer` class, imported by all notebooks.
- Embeddings and attention helpers are re-defined inline in each notebook (self-contained cells).
