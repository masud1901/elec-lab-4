# Running the workshop experiments on Kaggle

The scripts are plain Python and work unchanged in a Kaggle notebook. All
experiments are driven by `Debiasing/experiments.py`; `Debiasing/main.py`
still produces the three core runs exactly as `docs/instruction.md` describes.

## 1. Get the code onto Kaggle

Three options, pick one:

**A. Upload as a private dataset (recommended for repeat runs)**

1. Zip the `Debiasing/` folder.
2. Kaggle → *Datasets* → *New Dataset* → upload the zip → create.
3. In the notebook: *Add Data* → your dataset, then:

```python
!mkdir -p /kaggle/working/work
!unzip -q -o /kaggle/input/<your-dataset-slug>/Debiasing.zip -d /kaggle/working/work
%cd /kaggle/working/work/Debiasing
```

**B. `%%writefile` cells** — paste each `.py` file into a cell that starts with
`%%writefile data.py` etc. (one cell per file), then `%cd /kaggle/working`.

**C. Git clone** — if you push the folder to a repo, `!git clone <url>` in the
notebook and `%cd` into `.../Debiasing`.

## 2. Notebook settings

- **Internet: ON** (Settings → Internet). The loader downloads UCI Adult from
  OpenML the first time. If OpenML is unreachable it falls back automatically
  to `kagglehub` (`uciml/adult-census-income`) and then to the raw UCI CSVs.
- **Accelerator: GPU T4 x2 or P100** is plenty. CPU-only also works; the whole
  matrix is small.
- No extra installs are required (`torch`, `sklearn`, `pandas`,
  `matplotlib` and `kagglehub` are preinstalled).

## 3. Smoke test first (~1 min)

```python
!python experiments.py --only core,q6,q9,q10,q11,q12 --epochs 1 --out /kaggle/working/runs_smoke
```

This exercises every code path (baseline, parity, eq-odds, hidden adversary,
`eq_opp`, gradient reversal, hidden-layer input, threshold tuning). It writes
throw-away models to `runs_smoke/`, so delete that folder afterwards.

## 4. Full run (~10-30 min on GPU)

```python
!python experiments.py --out /kaggle/working/runs
```

- Runs: 3 core (sex) + Q6 (2) + Q7 (1) + Q8 (3, race) + Q9 (1) + Q10 (1) +
  Q11 (1) = **12 trainings**, plus Q1-Q5 and Q12 which reuse cached models.
- Every run is cached in `runs/<tag>/` (`model.pt`, `history.csv`,
  `results.json`, `curves.png`). Re-running skips finished runs; add `--force`
  to retrain.
- Individual questions: `--only q2,q6` (any of `core,q1,...,q12,summary`).

## 5. Look at the results

```python
import pandas as pd
from IPython.display import Image, display

print(pd.read_csv("/kaggle/working/runs/summary.csv").round(4).to_string())

for name in ["scores.png", "loss_A.png", "q5_correlations.png", "q6_stability.png",
             "q7_alpha.png", "q8_race_vs_sex.png", "q9_eqopp.png", "q10_grl.png",
             "q11_hiddenfeat.png", "q12_thresholds.png"]:
    display(Image(f"/kaggle/working/work/Debiasing/{name}"))
```

Figures without a `qN_` prefix (`scores.png`, `loss_A.png`) are the ones the
instruction asks for by name; the rest are numbered by question.

## 6. Take the outputs away

```python
!cd /kaggle/working && zip -qr workshop_results.zip work/Debiasing/runs \
    work/Debiasing/*.png work/Debiasing/docs
```

Download `workshop_results.zip` from the notebook's *Output* panel, or commit
`runs/summary.csv` and the PNGs into the repo.

## 7. Publishing the report

`docs/answers.md` is the draft report. After the Kaggle run, copy the PNGs and
`runs/summary.csv` into `docs/figures/` and `docs/` (or keep the `runs/`
directory), update the numbers, and export the Markdown to PDF if needed.

## Notes / gotchas

- **First run downloads the dataset.** OpenML Adult v2 is ~4 MB; the split is
  deterministic (`SEED=42`) and 70/15/15 stratified on the four `(y, z)` cells.
- **`scores.png` is saved at the repo root** (as the instruction says) and a
  copy is kept in `runs/q2/`.
- **Two runs share the same seed/initialisation**, so differences between
  configurations reflect the method, not the seed. To measure seed stability
  instead, call `train(..., seed=...)` with different seeds in a scratch cell.
- **`alpha_schedule="sqrt"`** uses `alpha_t = alpha * sqrt(t / T)` (0.22 → 1.0
  over training). A raw `alpha * sqrt(t)` reaches 4.5x by epoch 20 and is left
  as an optional ablation.
- **Q12 never touches the test set for tuning**: thresholds are chosen on the
  validation split, then evaluated once on test.
