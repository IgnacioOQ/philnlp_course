# Transformer Tutorial — Claude Context

## Environment

- **Python:** `.venv` in project root, Python 3.13.7
- **Packages:** numpy 2.4.4, matplotlib 3.10.9, notebook 7.5.5 / JupyterLab 4.5.6
- **Registered kernel:** `transformer-tutorial`, display name `.venv (3.13.7)`
- **Do NOT use** the system Anaconda Python (`/Users/ignacio/anaconda3`) — it has a broken PyTorch install that crashes the Jupyter kernel.

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
| `04_multihead_attention.ipynb` | 5 — Multi-Head Attention | ⬜ Next |
| `05_positional_encoding.ipynb` | 6 — Positional Encodings | ⬜ |
| `06_feedforward.ipynb` | 7 — Feed-Forward + Layer Norm | ⬜ |
| `07_encoder.ipynb` | 9–10 — Encoder Block + Stack | ⬜ |
| `08_transformer.ipynb` | 12–13 — Full PyTorch Transformer + Training | ⬜ |

## Shared utilities

- `tokenizer.py` — `CharTokenizer` class, imported by all notebooks.
- Embeddings and attention helpers are re-defined inline in each notebook (self-contained cells).
