# Span Masking: Boundary Screening and Radial Conservation

Replication package for a study of span masking in JEPA-style language
representation learning. The repository contains:

- the exact training/experiment script that produced every run,
- the full set of 49 run records (JSON) and the aggregated summary table,
- the numerical verification scripts for every theorem and proposition in
  the paper, run independently of the training pipeline, and
- the code that generates every figure in the paper directly from the
  verification scripts' output and from the run records.

No number in the paper is hand-entered; everything traces to one of the
scripts in this repository.

## Contents

```
p1_experiment.py       canonical training/experiment runner
experiment_spec.md     frozen experimental design and decision rules
design_p1.py           matched-k design with precomputed effect sizes

verification/          numerical checks for every theorem and proposition,
                        independent of the training pipeline
  verify_thm1.py            Theorem 1 (boundary screening): Gaussian sweep
                             and an exhaustive discrete-chain enumeration
  verify_unbounded.py       Proposition 4 (unbounded objective)
  canonical_thresholds.py   single source for every collapse threshold
                             quoted anywhere in the paper
  jepa_linear.py            linear-JEPA pilot used to test the spectral
                             regularisation predictions
  bridge_experiment.py      English character-level information estimates
  bridge_control2.py        context-width control for the above
  check_split.py            interleaved vs. contiguous train/test split,
                             used to rule out a split artefact
  sweep_eta.py, test_anisotropy.py, test_eps_threshold.py, test_thm3.py
                             supporting sweeps for the epsilon-threshold law
  audit_*.py                 robustness audits run after the headline
                             results, checking assumptions the main
                             analysis depends on (single-seed sensitivity,
                             untested parameter ranges, implementation
                             correctness)
  canonical_lowpressure.txt  low-pressure equilibrium reference values

figures/                figure-generation code and its output
  make_figures.py            reads verification-script output and the
                              run records directly; produces every main
                              figure as vector PDF and 600 dpi PNG
  make_equations.py          renders the paper's display equations as
                              images (see note below)
  fig1_*.pdf/.png ... fig6_*.pdf/.png

results/                 the 49 run records this study reports
  <run_id>.json              one file per run: config, per-step
                              diagnostics, and final probe scores
  summary.csv                 aggregated table over all runs
```

## Reproducing a run

```
pip install -r requirements.txt
python p1_experiment.py selftest   # numpy-only sanity check, seconds
python p1_experiment.py time       # measures real per-step time on this
                                    # machine and recomputes the full-run
                                    # estimate; trust this over any number
                                    # written in experiment_spec.md
python p1_experiment.py all        # executes the full run ledger; safe to
                                    # interrupt, finished runs are skipped
                                    # on restart
python p1_experiment.py aggregate  # writes results/summary.csv
```

A GPU is strongly recommended; `all` will run on CPU but is impractically
slow for anything beyond `selftest`.

## Reproducing the verification results

Each script under `verification/` is standalone and can be run directly,
e.g. `python verification/verify_thm1.py`. None of them depend on
`p1_experiment.py` having been run first; they check the theory
independently, mostly against closed-form and exhaustively-enumerated
cases rather than against the trained-model results.

## Reproducing the figures

```
cd figures
python make_figures.py     # figures 1-6
python make_equations.py   # display-equation images used in the paper
```

`make_figures.py` reads directly from `verification/` script output and
from `results/summary.csv` / the individual run JSON files; it does not
take any number as a hand-entered constant.

Note on `make_equations.py`: figures and equations are rendered to image
files rather than produced as native word-processor equation objects,
because the equation objects could not be reliably re-verified visually in
the tooling used to prepare this manuscript. Rendering to an image kept
every equation checkable by eye against the source LaTeX-style expression
before it was used in the paper.

## Data

The pretraining corpus (WikiText-103) and the probe datasets (AG News,
SST-2, and the Brown corpus) are public and are downloaded by identifier
inside `p1_experiment.py`; no dataset files are stored in this repository.

## License

Code and data in this repository are released under the MIT License (see
`LICENSE`). If a different license better suits your intended use, please
open an issue.
