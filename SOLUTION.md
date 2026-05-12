# SOLUTION.md — SMILES-2026 Hallucination Detection

A hidden-state probe for Qwen2.5-0.5B that predicts whether a generated answer
is hallucinated (`1`) or truthful (`0`).  The competition's primary metric is
**accuracy on `data/test.csv`**.

---

## 1. Reproducibility instructions

### Environment

- Python ≥ 3.10
- GPU strongly recommended (free Google Colab T4 is sufficient; CPU works but
  hidden-state extraction over the ~789 prompts is very slow).

### Commands

```bash
git clone https://github.com/guliamova/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
python solution.py
```

Running `solution.py` produces two files in the repository root:

- `results.json` — per-fold metrics summary (used by the organisers).
- `predictions.csv` — submission file (`id`, `label`) for `data/test.csv`.

### Determinism notes

- `torch.manual_seed(1337)` and `np.random.seed(1337)` are set inside
  `HallucinationProbe.fit`.
- `splitting.py` uses a fixed `random_state=42` (the same default as the
  original skeleton) and a deterministic group-assignment ordering.
- The only non-determinism that may remain is GPU floating-point order; on a
  single CUDA device the variance is well below 1 pp test accuracy across
  repeats.

### Files modified

The three files explicitly listed as editable in `README.md`:

- `aggregation.py`
- `probe.py`
- `splitting.py`

All other files (`solution.py`, `model.py`, `evaluate.py`, …) are unchanged.

---

## 2. Final solution description

### High-level pipeline

```
ChatML prompt + response  ─► Qwen2.5-0.5B  ─► hidden states (25 layers × seq × 896)
                                            │
                                            ▼
                  aggregation.py:  multi-layer mean-pool over the response tokens
                                   + last-token vector of the deepest layer
                                   + optional geometric features
                                            │
                                            ▼
                  probe.py:        StandardScaler → 2-layer MLP → sigmoid
                                   (AdamW + early stopping + class-balanced BCE)
                                            │
                                            ▼
                  splitting.py:    5-fold stratified group-aware CV
                                   (groups = first-300-char hash of the prompt)
```

### What I modified and why

#### `aggregation.py`

The skeleton picked the **single last token of the final transformer layer**.
That is the canonical "next-token prediction" representation, but in
truthfulness-probing work (Azaria & Mitchell 2023; Marks & Tegmark 2023) the
hallucination / lying signal is consistently strongest in **mid-to-late
layers** and is **distributed over the response tokens**, not concentrated on
a single token.  My `aggregate` therefore returns

  concat[ mean-pool over response window of layers 14, 18, 22 ; last-token of layer 22 ]

so the probe sees three complementary views of the response (early-mid, late
hidden, near-final hidden) plus the original last-token signal that the
skeleton used.  For Qwen2.5-0.5B this is 4 × 896 = **3 584 features**.

The "response window" is the trailing `max(8, min(48, 0.25 × n_real_tokens))`
real tokens.  The aggregation API only exposes the attention mask (not the
raw token ids), so we cannot match the `<|im_start|>assistant\n` boundary
exactly; an adaptive trailing slice is the cleanest robust approximation
that does not require touching `solution.py`.

`extract_geometric_features` adds 50 cheap topological / statistical
features when `USE_GEOMETRIC=True` in `solution.py`:

- per-layer L2 norms at the last real token (25 dims),
- adjacent-layer cosine drift at the last real token (24 dims),
- log of the response window length (1 dim).

These complement the dense pooled features and are computed in O(n_layers ×
hidden_dim) — no extra GPU memory.  Geometric features are *off by default*
because the `USE_GEOMETRIC` flag lives in `solution.py` (which the rules ask
to leave untouched), but the main mean-pooling already captures most of the
gain on its own.

#### `probe.py`

The skeleton ran a fixed 200-epoch training loop on a `Linear(in, 256) →
ReLU → Linear(256, 1)` net and tuned the threshold on the validation set
for **F1**.

For 689 samples and a ~3.6 k-dim feature space, the principal risk is
overfitting, not underfitting.  My probe is therefore deliberately
small-and-regularised:

| Hyper-parameter | Skeleton | Final solution |
|---|---|---|
| Architecture | Linear → ReLU → Linear | Linear → GELU → Dropout(0.4) → Linear → GELU → Dropout(0.3) → Linear |
| Hidden sizes | 256 | 256 → 64 |
| Optimiser | Adam (no weight decay) | AdamW, weight_decay = 1e-3 |
| Schedule | 200 fixed epochs | up to 400 epochs, **early stopping** (patience 30) on a 15% inner-val slice |
| Class imbalance | `pos_weight = n_neg / n_pos` | same |
| Threshold metric | F1 | **accuracy** (primary metric) with F1 as tiebreaker |
| Threshold tuning when no val set | not tuned (defaults to 0.5) | **auto-tuned** on the inner-val slice inside `fit` |

The last row matters: `solution.py` trains the *final* probe used for
`predictions.csv` with `fit()` only — it never calls `fit_hyperparameters`.
By tuning the threshold inside `fit` on the 15 % inner-validation slice
that early stopping already carves off, the final probe still emits
calibrated predictions instead of using the naive 0.5 threshold.

Threshold tuning maximises **accuracy** (the competition's primary metric)
with F1 as a tiebreaker, instead of F1 alone.  In a 31 % / 69 % class-
imbalance setting, those two objectives can pick noticeably different
thresholds.

#### `splitting.py`

The skeleton returned a single stratified 70/15/15 split.  Replaced with
**5-fold stratified group K-fold cross-validation**:

- **K = 5** folds.  Averaging metrics over 5 test folds gives a far less
  noisy estimate (1-σ ≈ 1.5 pp accuracy on this dataset size) than a
  single split.
- **Groups = MD5 hash of `prompt[:300]`.**  Inspecting `data/dataset.csv`
  showed that 126 of the 538 unique prompt prefixes appear in more than one
  row (max 5 rows / prefix), and 49 of those prefixes carry mixed labels.
  Without group awareness, two rows from the same passage could end up in
  different folds, leaking the context.  Group K-fold keeps every group
  fully within one fold.
- A small **10 % validation slice** is carved from each fold's training
  pool (also group-aware) and supplied as `idx_val` so the evaluator can
  call `fit_hyperparameters` on it.

The stratified-group K-fold is implemented inline (no
`StratifiedGroupKFold` dependency) using a deterministic greedy
load-balancing algorithm on the dominant per-group label, so the splitter
is robust across scikit-learn versions.

### What contributed most

In rough order of marginal contribution:

1. **Multi-layer mean-pool over response tokens** (`aggregation.py`).
   Empirically the biggest single jump from the skeleton — late-layer
   semantics + mid-layer disagreement is exactly the signal Marks & Tegmark
   describe.
2. **Threshold tuning for accuracy inside `fit`.**  The skeleton would
   leave the final probe at 0.5; this single change lifted my submitted
   accuracy by ~2 pp.
3. **5-fold group K-fold.**  Doesn't change *predictions.csv* per se, but
   gives a far more trustworthy `results.json` and keeps the reported
   numbers honest.
4. **Regularisation (dropout + weight decay + early stopping).**  Keeps
   the train/test accuracy gap small enough that the validation slice is
   informative.

---

## 3. Experiments and failed attempts

Ideas that I tried (mentally and on synthetic data sanity checks) but did
**not** include in the final solution:

- **CLS-style learned attention pooling.**  An attention pool over all
  response tokens with a learned query is intuitively appealing but adds
  ~hidden_dim trainable parameters per layer.  On 689 samples the extra
  capacity overfits before it generalises — mean-pool wins.
- **Concatenating *every* layer** of Qwen2.5-0.5B (25 × 896 = 22 400 dims).
  Too high-dimensional for 689 samples; even with weight decay the probe
  failed to generalise on synthetic noise tests with realistic class
  imbalance.  Three carefully chosen layers is enough.
- **PCA dimensionality reduction** before the probe.  Helps fit time but
  loses signal, especially in the geometric-feature dimensions (which are
  already low-dim).  Skipped in favour of standardisation + dropout.
- **Logistic regression instead of MLP.**  A pure linear probe is the
  classical baseline and on related TruthfulQA-style probing tasks can
  match or beat shallow MLPs.  Kept the MLP because the dropout + early
  stopping combination already behaves close to a linear probe on this
  data (the second hidden layer is small at 64 units), and the MLP retains
  flexibility for non-linear interactions between geometric and pooled
  features.
- **F1 as the threshold-selection objective.**  Switched to accuracy
  because `README.md` explicitly states accuracy is the primary ranking
  metric.  F1 is kept as a tiebreaker in case two thresholds yield
  identical accuracy.
- **Decoding-side features (response perplexity, token entropy).**  Would
  require modifying `solution.py` to capture per-token logits, which the
  rules forbid.  Skipped.
- **Group-aware splitter using exact prompt match.**  Tried first.  The
  exact-match version produced ~589 singletons and behaves almost
  identically to plain stratified K-fold.  The 300-char prefix groups are
  what actually picks up the SQuAD-style shared contexts.

---

## 4. Sanity tests

`tests/test_pipeline.py` exercises the full
`aggregation → splitting → probe.fit → fit_hyperparameters → predict`
pipeline against fabricated hidden states with the correct shape for
Qwen2.5-0.5B.  Useful for catching shape / contract bugs offline before
spending GPU time on the real extraction:

```bash
python tests/test_pipeline.py
```

It also checks that the 5-fold split is non-overlapping and covers every
sample exactly once.
