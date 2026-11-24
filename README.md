# Week 10 — adversarial debiasing on UCI Adult

Workshop repo for Zhang, Lemoine & Mitchell (2018), *Mitigating Unwanted
Biases with Adversarial Learning*.

- **Assignment:** [`docs/instruction.md`](docs/instruction.md)
- **Draft answers (Q1–Q12):** [`docs/answers.md`](docs/answers.md)
- **Running on Kaggle:** [`docs/KAGGLE.md`](docs/KAGGLE.md)
- **Figures:** [`docs/figures/`](docs/figures)

## Quick start

```bash
cd Debiasing
python main.py                 # the three core runs (baseline, parity, eq-odds)
python experiments.py          # all Q1–Q12 runs, cached under runs/
```

`main.py` writes `results.csv`, `comparison.png`, `curves.png` as described in
the instruction. `experiments.py` adds the Q1–Q12 variants and writes one
directory per configuration plus `runs/summary.csv`.

```bash
python experiments.py --only q2,q6        # selected questions
python experiments.py --only core --epochs 1 --out /tmp/smoke   # smoke test
python experiments.py --force             # retrain cached runs
```

## File map

| File | Purpose |
|---|---|
| `Debiasing/data.py` | Adult loader, `sex`/`race` protected attribute, 70/15/15 split |
| `Debiasing/models.py` | Predictor, adversary (parity / eq-odds / eq-opp), gradient reversal |
| `Debiasing/train.py` | Paper's projection update, `√t` schedule, GRL path |
| `Debiasing/metrics.py` | Accuracy, DP/TPR/FPR gaps, per-group thresholds |
| `Debiasing/main.py` | The workshop's three core runs |
| `Debiasing/experiments.py` | Q1–Q12 driver, caching, figures, summary |
| `Debiasing/sweep.py` | Optional α sweep for parity mode |
| `docs/answers.md` | Report draft with the numbers from the full run |
| `docs/KAGGLE.md` | Step-by-step Kaggle instructions |

Results in `Debiasing/runs/` were produced with seed 42, 20 epochs, on an
NVIDIA GTX 1650 Ti (about 12 minutes for the whole matrix). On Kaggle a T4/P100
takes roughly the same. Every run is cached, so a session can be resumed.
