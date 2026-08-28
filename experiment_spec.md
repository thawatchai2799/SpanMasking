# Experiment Specification (v1)

Everything below is sized for one consumer GPU (RTX 3060-class or a free
Kaggle/Colab T4) inside a 24-hour budget, with margin. Every number that a
result will depend on is fixed here, before any training run, so that no
quantity can drift between configurations the way the collapse thresholds once
did.

## 1. Model (identical across all cells)

| component | value | reason |
|---|---|---|
| encoder | Transformer, 4 layers, d=192, 4 heads, FFN 512, pre-LN | ~4.5M params incl. embeddings; one run ≈ 30–35 min |
| sequence length | T = 128 | matches the design grid |
| mask sampling | k blocks of exact length l, positions uniform at random without overlap or touching, resampled per sequence per step | the design fixes (k, l); deterministic placement would confound position with cell |
| tokenizer | BPE, 8k merges, trained once on the pretraining corpus | shared across every run; vocab is not a variable |
| predictor | 2-layer MLP 192→192, GELU | final layer linear, so Theorem 2 applies to it |
| target encoder | EMA of context encoder, τ = 0.996 | standard JEPA |
| objective | cosine on masked positions | one objective for P1; regularisers appear only in the Section-6 arm |
| optimizer | AdamW, lr 3e-4, wd ω = 0.04, cosine schedule, warmup 500 | |
| batch / steps | 64 × 9,000 (≈ 74M tokens) | cell A runs 18,000 steps in its equalised variant (~147M tokens = 1.8 epochs of the 80M-token corpus; data repeats and we say so) |
| clipping | global-norm 1.0 everywhere EXCEPT the two P3 runs | P3 is unreadable under clipping (Section 3.4) |
| P3 exception | the two P3 runs use plain SGD + weight decay, no momentum, no clipping | Theorem 2 assumes gradient flow; under AdamW the measured decay deviates 7.8% (SGD: 0.02%), so an AdamW run cannot test it |

## 2. Data

Pretraining: WikiText-103, first 80M tokens, fixed shuffle seed 0.
Held out: the standard validation split, never trained on.

## 3. Probes (the part P2's refutation forces us to get right)

P2 failed in the linear setting under a *reconstruction* probe, which a
low-rank representation can serve well. The nonlinear rerun must therefore
separate probe types explicitly. All probes use frozen features; the probe is
a single linear layer; 5-fold CV on the probe's own split; identical feature
extraction (mean-pool over non-pad positions for sequence tasks, per-token
states for token tasks).

**Semantic probes** (the P1 endpoint):
- AG News topic (4-way): a fixed 20k subsample of the 120k train set
  (probe-fitting speed), full 7.6k test set
- SST-2 sentiment: GLUE train for probe fitting, the 872-example
  validation set as the test set (GLUE test labels are hidden)

All probe features are extracted from **unmasked** inputs; no [MASK] token
ever appears at probe time. This matters for P4 in particular: probing the
state at a masked position for the identity of the missing token is literally
the MLM training objective, and the MLM arm would win it by construction. The
lexical probes below are therefore *word-content* probes on clean text.

**Lexical probes** (the P4 endpoint):
- POS tagging, universal tagset, on Brown corpus sentences re-tokenized with
  our BPE; per-token linear probe, first-subtoken convention
- word content: from the contextual state at position t of an unmasked
  sentence, predict the token id at t (top-1k vocabulary). MLM-trained
  models may still preserve more token identity — that is the substantive,
  non-circular version of the prediction

**Baselines, run once and reused everywhere:**
- random-init frozen encoder (same architecture, no training)
- TF-IDF + logistic regression on the raw text (semantic tasks only)

A pretrained cell "passes" only relative to these: if random-init matches it,
the pretraining did nothing and no P1 comparison is meaningful. This gate is
checked before any P1 statistics are computed.

## 4. Run ledger

| block | cells | seeds | steps | runs | GPU-min @33min |
|---|---|---|---|---|---|
| P1 grid | A, B, D | 5 | 9k | 15 | 495 |
| A equalised | A@18k | 5 | 18k | 5 | 330 |
| P4 MLM arm | B + aux MLM (α=0.1) | 5 | 9k | 5 | 165 |
| P3 no-clip | B, plain SGD, clipping off, ω ∈ {0.04, 0.2} | 1 | 9k | 2 | 66 |
| Section-6 negative-result arm | B + VICReg(25,1), B + LDB(β=.01, ε=1e-4, κ=1) | 3 | 9k | 6 | 198 |
| random-init baseline | — | 1 | 0 | 1 | 5 |
| **total** | | | | **34** | **1,259 min ≈ 21.0 h** |

The 33 min/run figure is an unmeasured, deliberately padded estimate — a
4.5M-parameter model at batch 64 is bandwidth- and overhead-bound and may run
3–5× faster on a T4, in which case the freed budget goes to the P1b sweep
(Section 6) before anything else. The first action on the target machine is a
200-step timing run to replace this estimate; the ledger is recomputed from
the measured number before any full run starts. Probing adds ≈ 40 min total
(CPU-parallel with training). Margin ≈ 3 h for
re-runs. If anything overruns, the drop order is: P3 second ω, then one seed
from the Section-6 arm — never a P1 seed.

## 5. Decision rules, fixed in advance

- **Gate**: every trained cell must beat random-init on both semantic probes
  by ≥ 2 points (mean over seeds); otherwise stop and report the failure —
  no P1 statistics on features indistinguishable from noise.
- **P1**: AG News is the primary endpoint and SST-2 secondary, fixed here so
  no post-hoc choice between them is possible. Compare |acc(A) − acc(B)|
  against |acc(B) − acc(D)| per seed; Wilcoxon signed-rank across seeds,
  one-sided. With 5 seeds the smallest attainable one-sided p is 1/32 ≈ 0.031,
  so significance requires a unanimous ordering; we state this power limit in
  the paper rather than discovering it in review. Report both equalised and
  unequalised A. Prediction fails if the matched-k gap is larger in both.
- **P4**: MLM arm vs plain B. Δlexical is the mean delta over the two
  lexical probes and Δsemantic the mean over the two semantic probes, fixed
  here so no per-probe selection is possible. Three outcomes, fixed now: **supported** if
  Δlexical > 0 and Δsemantic ≤ 0; **refuted** if Δsemantic > Δlexical;
  **indeterminate** otherwise (e.g. both improve with lexical ahead), reported
  as such without reinterpretation.
- **P3**: fit exp decay to ‖W‖_F of the predictor's final layer in the
  no-clip runs; report fitted rate against ω with no tolerance chosen
  post hoc — the two-ω design exists so the rate can be shown to *track* ω,
  not merely to be small.
- **Section 6**: report probe deltas of VICReg/LDB arms against B with CIs;
  any conclusion of "no benefit" requires the CI to exclude a 1-point gain.
- **Diagnostics logged every 250 steps, all runs**: λ_min and effective rank
  of the predictor-output covariance (2,000-sample fixed batch), ‖W‖_F,
  train loss. One canonical script writes every number; no value in the
  paper may come from anywhere else.

## 6. What this spec does not cover

Tokenizer-level depth calibration (Section 3.5's character-level rates do not
transfer and we do not pretend otherwise); any claim about optimal span
length (P1b is stated-not-tested unless margin remains); and any comparison
of LDB against a *tuned* VICReg, which Section 3.3 shows we cannot make
fairly.
