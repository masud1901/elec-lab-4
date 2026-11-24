
"""Workshop experiment driver: Q1-Q12 from docs/instruction.md.

Every run is cached under ``runs/<tag>/`` (model.pt, history.csv, results.json,
curves.png), so re-running is cheap and Kaggle sessions are resumable.

Usage
-----
    python experiments.py                 # everything (Q0 core + Q1..Q12)
    python experiments.py --only q2,q6    # selected questions
    python experiments.py --only core --epochs 2    # quick smoke test
    python experiments.py --force         # retrain cached runs
"""

import argparse
import json
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")                      # save figures to files (works with !python)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import load_adult
from metrics import evaluate, evaluate_probs, evaluate_thresholds, predict_proba
from models import Predictor
from train import train

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "runs")
EPOCHS = 20
DATA_CACHE, RUN_CACHE = {}, {}


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
def get_data(protected="sex", verbose=True):
    if protected not in DATA_CACHE:
        DATA_CACHE[protected] = load_adult(protected=protected, verbose=verbose)
    return DATA_CACHE[protected]


def tag_dir(tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    return d


def save_curves(hist, tag, d):
    cols = ["loss_A", "accuracy", "DP gap", "TPR gap", "FPR gap"]
    fig, axes = plt.subplots(1, len(cols), figsize=(22, 4))
    for ax, c in zip(axes, cols):
        ax.plot(hist["epoch"], hist[c], marker="o", ms=3)
        ax.set_title(f"{c}  ({tag})")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(d, "curves.png"), dpi=110)
    plt.close(fig)


def get_run(tag, protected="sex", train_kwargs=None, force=False, epochs=EPOCHS,
            verbose=True):
    """Train (or load) one configuration; returns (predictor, history, metrics)."""
    key = (tag, protected)
    if key in RUN_CACHE and not force:
        return RUN_CACHE[key]

    data = get_data(protected, verbose=verbose)
    d = tag_dir(tag)
    mp = os.path.join(d, "model.pt")
    hp = os.path.join(d, "history.csv")
    rp = os.path.join(d, "results.json")
    P = Predictor(data["n_features"]).to(data["device"])

    if os.path.exists(mp) and not force:
        P.load_state_dict(torch.load(mp, map_location=data["device"], weights_only=True))
        hist = pd.read_csv(hp)
        res = json.load(open(rp))
    else:
        P, hist = train(data, epochs=epochs, verbose=verbose, tag=tag, **(train_kwargs or {}))
        torch.save(P.state_dict(), mp)
        hist.to_csv(hp, index=False)
        res = evaluate(P, data["X_test"], data["y_test"], data["z_test"])
        json.dump(res, open(rp, "w"), indent=2)
        save_curves(hist, tag, d)

    RUN_CACHE[key] = (P, hist, res)
    return RUN_CACHE[key]


def core(force=False, epochs=EPOCHS):
    return {
        "baseline": get_run("sex_baseline", "sex", {"debias": False}, force, epochs),
        "parity": get_run("sex_parity", "sex", {"mode": "parity", "alpha": 1.0}, force, epochs),
        "eq_odds": get_run("sex_eq_odds", "sex", {"mode": "eq_odds", "alpha": 1.0}, force, epochs),
    }


CORE_KWARGS = {
    "baseline": {"debias": False},
    "parity": {"mode": "parity", "alpha": 1.0},
    "eq_odds": {"mode": "eq_odds", "alpha": 1.0},
}


def race_core(force=False, epochs=EPOCHS):
    return {k: get_run(f"race_{k}", "race", v, force, epochs) for k, v in CORE_KWARGS.items()}


def table(runs, title):
    rows = {name: res for name, (_, _, res) in runs.items()}
    df = pd.DataFrame(rows).T
    print(f"\n=== {title} ===")
    print(df.round(4).to_string())
    return df


def group_rates(p, y, z, thr=0.5):
    yhat = (np.asarray(p) >= thr).astype(float)
    out = {}
    for name, val in (("M", 1), ("F", 0)):
        m = z == val
        out[f"sel_{name}"] = float(yhat[m].mean())
        out[f"TPR_{name}"] = float(yhat[m & (y == 1)].mean())
        out[f"FPR_{name}"] = float(yhat[m & (y == 0)].mean())
    return out


def stability(hist, col, last=10):
    return {"mean": float(hist[col].tail(last).mean()), "std": float(hist[col].tail(last).std())}


def binary_entropy_nats(p):
    p = float(p)
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def savefig(fig, name, subdir=None):
    if subdir:
        d = tag_dir(subdir)
        fig.savefig(os.path.join(d, name), dpi=120, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT, name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# --------------------------------------------------------------------------- #
# Q1 - adversary loss during training
# --------------------------------------------------------------------------- #
def q1(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    data = get_data("sex", verbose=False)
    Xtr, ytr, ztr = [t.numpy() for t in data["train_ds"].tensors]
    h_z = binary_entropy_nats(ztr.mean())
    h_z_given_y = sum((ytr == v).mean() * binary_entropy_nats(ztr[ytr == v].mean())
                      for v in (0.0, 1.0))

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("parity", "Parity (adversary sees yhat)"),
                       ("eq_odds", "Equalized odds (sees yhat and y)")):
        h = runs[key][1]
        ax.plot(h["epoch"], h["loss_A"], marker="o", ms=3, label=label)
        print(f"Q1 {key:8s}: L_A first={h['loss_A'].iloc[0]:.4f} "
              f"last={h['loss_A'].iloc[-1]:.4f} min={h['loss_A'].min():.4f} "
              f"last10 std={h['loss_A'].tail(10).std():.4f}")
    ax.axhline(h_z, ls="--", c="gray", label=f"H(z) = {h_z:.3f}")
    ax.axhline(h_z_given_y, ls=":", c="black", label=f"H(z|y) = {h_z_given_y:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("adversary BCE (nats)")
    ax.set_title("Q1: adversary loss by mode")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "loss_A.png", "q1")


# --------------------------------------------------------------------------- #
# Q2 - score histograms
# --------------------------------------------------------------------------- #
def q2(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    data = get_data("sex", verbose=False)
    z = data["z_test"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, key, title in ((axes[0], "baseline", "Baseline"),
                           (axes[1], "parity", "Debiased: parity")):
        p = predict_proba(runs[key][0], data["X_test"])
        ax.hist(p[z == 1], bins=bins, alpha=0.55, density=True, label="Men")
        ax.hist(p[z == 0], bins=bins, alpha=0.55, density=True, label="Women")
        ax.set_title(f"{title}\nmean score  M={p[z == 1].mean():.3f}  F={p[z == 0].mean():.3f}")
        ax.set_xlabel("sigmoid(P(x))"); ax.legend(); ax.grid(alpha=0.3)
        print(f"Q2 {title:10s}: mean M={p[z == 1].mean():.3f} F={p[z == 0].mean():.3f} "
              f"| P(>0.5) M={(p[z == 1] >= .5).mean():.3f} F={(p[z == 0] >= .5).mean():.3f}")
    axes[0].set_ylabel("density")
    plt.tight_layout()
    savefig(fig, "scores.png", "q2")


# --------------------------------------------------------------------------- #
# Q3 / Q4 - analysis of gaps
# --------------------------------------------------------------------------- #
def q3(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    data = get_data("sex", verbose=False)
    z = data["z_test"].numpy(); y = data["y_test"].numpy()
    print("\n=== Q3: per-group rates on the test set ===")
    for key in ("baseline", "parity", "eq_odds"):
        p = predict_proba(runs[key][0], data["X_test"])
        g = group_rates(p, y, z)
        print(f"{key:9s} sel M={g['sel_M']:.3f} F={g['sel_F']:.3f} | "
              f"TPR M={g['TPR_M']:.3f} F={g['TPR_F']:.3f} | "
              f"FPR M={g['FPR_M']:.3f} F={g['FPR_F']:.3f}")
    print(f"base rates: P(y=1|M)={y[z == 1].mean():.3f}  P(y=1|F)={y[z == 0].mean():.3f}")


def q4(force=False, epochs=EPOCHS):
    data = get_data("sex", verbose=False)
    Xtr, ytr, ztr = [t.numpy() for t in data["train_ds"].tensors]
    print("\n=== Q4: DP + EO feasibility ===")
    print(f"base rate gap = {abs(ytr[ztr == 1].mean() - ytr[ztr == 0].mean()):.3f} "
          f"(P(y=1|M)={ytr[ztr == 1].mean():.3f}, P(y=1|F)={ytr[ztr == 0].mean():.3f})")
    print("DP forces equal selection rates; EO forces equal TPR and FPR.")
    print("Together they force P(y=1|M) = P(y=1|F) unless the classifier is constant or perfect.")


# --------------------------------------------------------------------------- #
# Q5 - features correlated with sex
# --------------------------------------------------------------------------- #
def q5(force=False, epochs=EPOCHS):
    data = get_data("sex", verbose=False)
    Xtr, ytr, ztr = [t.numpy() for t in data["train_ds"].tensors]
    names = np.array(data["feature_names"])
    z = ztr.astype(float)
    r = np.array([np.corrcoef(Xtr[:, j], z)[0, 1] if Xtr[:, j].std() > 0 else 0.0
                  for j in range(Xtr.shape[1])])
    df = pd.DataFrame({"feature": names, "r_with_sex": r, "abs_r": np.abs(r)})
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)
    df["base_feature"] = df["feature"].str.split("=").str[0]
    grouped = (df.groupby("base_feature")["abs_r"].max().sort_values(ascending=False)
               .reset_index())

    d = tag_dir("q5")
    df.to_csv(os.path.join(d, "correlations.csv"), index=False)
    print("\n=== Q5: top 12 individual features correlated with sex ===")
    print(df.head(12).round(3).to_string(index=False))
    print("\n=== Q5: grouped by source column (max |r|) ===")
    print(grouped.head(12).round(3).to_string(index=False))

    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["r_with_sex"],
            color=np.where(top["r_with_sex"] > 0, "#2b6cb0", "#c05621"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("point-biserial correlation with z (Male=1)")
    ax.set_title("Q5: features most correlated with sex (after sex is removed)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q5_correlations.png")


# --------------------------------------------------------------------------- #
# Q6 - adversary with a hidden layer
# --------------------------------------------------------------------------- #
def q6(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    runs["parity_hiddenadv"] = get_run("sex_parity_hiddenadv", "sex",
                                       {"mode": "parity", "alpha": 1.0, "adv_hidden": 64},
                                       force, epochs)
    runs["eq_odds_hiddenadv"] = get_run("sex_eq_odds_hiddenadv", "sex",
                                        {"mode": "eq_odds", "alpha": 1.0, "adv_hidden": 64},
                                        force, epochs)
    print("\n=== Q6: single-layer vs hidden-layer adversary ===")
    for key in ("parity", "parity_hiddenadv", "eq_odds", "eq_odds_hiddenadv"):
        res, hist = runs[key][2], runs[key][1]
        dp, la = stability(hist, "DP gap"), stability(hist, "loss_A")
        print(f"{key:19s} acc={res['accuracy']:.4f} DP={res['DP gap']:.4f} "
              f"TPR={res['TPR gap']:.4f} FPR={res['FPR gap']:.4f} | "
              f"last10 DP {dp['mean']:.3f}±{dp['std']:.3f}  L_A {la['mean']:.3f}±{la['std']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, pair, title in ((axes[0], ("parity", "parity_hiddenadv"), "Parity"),
                            (axes[1], ("eq_odds", "eq_odds_hiddenadv"), "Equalized odds")):
        for key, label in zip(pair, ("1 layer", "hidden layer")):
            h = runs[key][1]
            ax.plot(h["epoch"], h["DP gap"], marker="o", ms=3, label=label)
        ax.set_title(f"Q6: DP gap, {title}")
        ax.set_xlabel("epoch"); ax.set_ylabel("DP gap"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q6_stability.png")


# --------------------------------------------------------------------------- #
# Q7 - adversary weight growing like sqrt(t)
# --------------------------------------------------------------------------- #
def q7(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    runs["parity_sqrt"] = get_run("sex_parity_sqrt", "sex",
                                  {"mode": "parity", "alpha": 1.0, "alpha_schedule": "sqrt"},
                                  force, epochs)
    print("\n=== Q7: constant alpha vs sqrt(t) schedule ===")
    for key in ("baseline", "parity", "parity_sqrt"):
        res = runs[key][2]
        print(f"{key:12s} acc={res['accuracy']:.4f} DP={res['DP gap']:.4f} "
              f"TPR={res['TPR gap']:.4f} FPR={res['FPR gap']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for key, label in (("baseline", "Baseline"), ("parity", "Parity const"),
                       ("parity_sqrt", "Parity sqrt(t)")):
        h = runs[key][1]
        axes[0].plot(h["epoch"], h["DP gap"], marker="o", ms=3, label=label)
        axes[1].plot(h["epoch"], h["accuracy"], marker="o", ms=3, label=label)
    axes[0].set_title("Q7: DP gap"); axes[1].set_title("Q7: accuracy")
    for ax in axes[:2]:
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    axes[2].plot(runs["parity_sqrt"][1]["epoch"], runs["parity_sqrt"][1]["alpha"], marker="o", ms=3)
    axes[2].set_title("Q7: alpha_t"); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q7_alpha.png")


# --------------------------------------------------------------------------- #
# Q8 - race instead of sex
# --------------------------------------------------------------------------- #
def q8(force=False, epochs=EPOCHS):
    sex = core(force, epochs)
    race = race_core(force, epochs)
    print("\n=== Q8: race scenario (z = White) ===")
    for key in ("baseline", "parity", "eq_odds"):
        r = race[key][2]
        print(f"race/{key:9s} acc={r['accuracy']:.4f} DP={r['DP gap']:.4f} "
              f"TPR={r['TPR gap']:.4f} FPR={r['FPR gap']:.4f}")
    print(f"baseline DP gap: sex={sex['baseline'][2]['DP gap']:.4f}  "
          f"race={race['baseline'][2]['DP gap']:.4f}")
    print(f"baseline TPR gap: sex={sex['baseline'][2]['TPR gap']:.4f}  "
          f"race={race['baseline'][2]['TPR gap']:.4f}")

    metrics = ["DP gap", "TPR gap", "FPR gap"]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [sex["baseline"][2][m] for m in metrics], w, label="sex")
    ax.bar(x + w / 2, [race["baseline"][2][m] for m in metrics], w, label="race")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Q8: baseline bias, sex vs race"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q8_race_vs_sex.png")


# --------------------------------------------------------------------------- #
# Q9 - equal opportunity (adversary loss on y=1 only)
# --------------------------------------------------------------------------- #
def q9(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    runs["eq_opp"] = get_run("sex_eq_opp", "sex", {"mode": "eq_opp", "alpha": 1.0},
                             force, epochs)
    print("\n=== Q9: eq_opp vs eq_odds ===")
    print(f"{'run':9s} {'acc':>7s} {'DP':>7s} {'TPR':>7s} {'FPR':>7s}")
    for key in ("eq_odds", "eq_opp"):
        r = runs[key][2]
        print(f"{key:9s} {r['accuracy']:7.4f} {r['DP gap']:7.4f} "
              f"{r['TPR gap']:7.4f} {r['FPR gap']:7.4f}")

    metrics = ["DP gap", "TPR gap", "FPR gap"]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [runs["eq_odds"][2][m] for m in metrics], w, label="eq_odds")
    ax.bar(x + w / 2, [runs["eq_opp"][2][m] for m in metrics], w, label="eq_opp")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Q9: equal opportunity vs equalized odds"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q9_eqopp.png")


# --------------------------------------------------------------------------- #
# Q10 - gradient reversal layer
# --------------------------------------------------------------------------- #
def q10(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    runs["grl"] = get_run("sex_grl", "sex",
                          {"mode": "parity", "alpha": 1.0, "use_grl": True},
                          force, epochs)
    print("\n=== Q10: gradient reversal vs baseline/parity ===")
    for key in ("baseline", "parity", "grl"):
        r = runs[key][2]
        print(f"{key:9s} acc={r['accuracy']:.4f} DP={r['DP gap']:.4f} "
              f"TPR={r['TPR gap']:.4f} FPR={r['FPR gap']:.4f}")

    metrics = ["accuracy", "DP gap", "TPR gap", "FPR gap"]
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(("baseline", "parity", "grl")):
        ax.bar(x + (i - 1) * w, [runs[key][2][m] for m in metrics], w, label=key)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Q10: gradient reversal (one backward)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q10_grl.png")


# --------------------------------------------------------------------------- #
# Q11 - adversary on the predictor's last hidden layer
# --------------------------------------------------------------------------- #
def q11(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    runs["parity_hiddenfeat"] = get_run("sex_parity_hiddenfeat", "sex",
                                        {"mode": "parity", "alpha": 1.0, "adv_input": "hidden"},
                                        force, epochs)
    print("\n=== Q11: adversary on yhat vs on the 64-d hidden layer ===")
    for key in ("parity", "parity_hiddenfeat"):
        r, h = runs[key][2], runs[key][1]
        dp, la = stability(h, "DP gap"), stability(h, "loss_A")
        print(f"{key:18s} acc={r['accuracy']:.4f} DP={r['DP gap']:.4f} "
              f"TPR={r['TPR gap']:.4f} FPR={r['FPR gap']:.4f} | "
              f"last10 DP {dp['mean']:.3f}±{dp['std']:.3f}  L_A {la['mean']:.3f}±{la['std']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for key, label in (("parity", "adversary on yhat"), ("parity_hiddenfeat", "adversary on h(64)")):
        h = runs[key][1]
        axes[0].plot(h["epoch"], h["DP gap"], marker="o", ms=3, label=label)
        axes[1].plot(h["epoch"], h["loss_A"], marker="o", ms=3, label=label)
    axes[0].set_title("Q11: DP gap"); axes[1].set_title("Q11: adversary loss")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q11_hiddenfeat.png")


# --------------------------------------------------------------------------- #
# Q12 - per-group thresholds on the baseline
# --------------------------------------------------------------------------- #
def q12(force=False, epochs=EPOCHS):
    runs = core(force, epochs)
    data = get_data("sex", verbose=False)
    P = runs["baseline"][0]
    p_val = predict_proba(P, data["X_val"])
    y_val, z_val = data["y_val"].numpy(), data["z_val"].numpy()

    best, fallback = None, None
    for target in np.linspace(0.02, 0.60, 60):
        thr_m = np.quantile(p_val[z_val == 1], 1 - target)
        thr_f = np.quantile(p_val[z_val == 0], 1 - target)
        r = evaluate_probs(p_val, y_val, z_val, thr_m, thr_f)
        cand = {"target": float(target), "thr_m": float(thr_m), "thr_f": float(thr_f),
                "val_acc": r["accuracy"], "val_dp": r["DP gap"]}
        if fallback is None or cand["val_dp"] < fallback["val_dp"]:
            fallback = cand
        if cand["val_dp"] <= 0.01 and (best is None or cand["val_acc"] > best["val_acc"]):
            best = cand
    best = best or fallback

    base = runs["baseline"][2]
    tuned = evaluate_thresholds(P, data["X_test"], data["y_test"], data["z_test"],
                                best["thr_m"], best["thr_f"])
    parity = runs["parity"][2]

    print("\n=== Q12: per-group thresholds (baseline, tuned on validation) ===")
    print(f"target positive rate={best['target']:.3f}  thr_M={best['thr_m']:.4f}  "
          f"thr_F={best['thr_f']:.4f}  val DP={best['val_dp']:.4f}  val acc={best['val_acc']:.4f}")
    rows = {"baseline @0.5": base,
            "baseline + group thresholds": tuned,
            "adversarial parity": parity}
    df = pd.DataFrame(rows).T
    print(df.round(4).to_string())
    df.to_csv(os.path.join(tag_dir("q12"), "results.csv"))

    metrics = ["accuracy", "DP gap", "TPR gap", "FPR gap"]
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, res) in enumerate(rows.items()):
        ax.bar(x + (i - 1) * w, [res[m] for m in metrics], w, label=name)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title("Q12: post-hoc thresholds vs adversarial parity"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "q12_thresholds.png")


# --------------------------------------------------------------------------- #
# Summary + CLI
# --------------------------------------------------------------------------- #
def summary():
    rows = []
    for path in sorted(os.listdir(OUT)):
        rp = os.path.join(OUT, path, "results.json")
        if os.path.exists(rp):
            rows.append({"run": path, **json.load(open(rp))})
    if not rows:
        print("No runs yet.")
        return
    df = pd.DataFrame(rows).set_index("run")
    df.to_csv(os.path.join(OUT, "summary.csv"))
    print("\n=== All runs (test set) ===")
    print(df.round(4).to_string())
    print(f"\nSaved {os.path.join(OUT, 'summary.csv')}")


QUESTIONS = {
    "core": lambda f, e: table(core(f, e), "Core 3 runs (sex)"),
    "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5, "q6": q6,
    "q7": q7, "q8": q8, "q9": q9, "q10": q10, "q11": q11, "q12": q12,
}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="'all', or comma-separated: core,q1,...,q12,summary")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--force", action="store_true", help="retrain cached runs")
    ap.add_argument("--out", default=OUT, help="output directory for runs/")
    args = ap.parse_args()
    OUT = os.path.abspath(args.out)
    os.makedirs(OUT, exist_ok=True)

    names = (["core"] + [f"q{i}" for i in range(1, 13)] + ["summary"]
             if args.only == "all" else [s.strip() for s in args.only.split(",")])
    for name in names:
        if name not in QUESTIONS and name != "summary":
            raise SystemExit(f"unknown experiment '{name}'")
        print(f"\n########## {name} ##########")
        QUESTIONS[name](args.force, args.epochs) if name != "summary" else summary()


if __name__ == "__main__":
    main()
