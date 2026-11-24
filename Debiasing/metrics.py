
import numpy as np
import torch


def _gaps(yhat, y, z):
    def rate(mask):
        return yhat[mask].mean() if mask.any() else np.nan

    return {
        "accuracy":  float((yhat == y).mean()),
        "DP gap":    float(abs(rate(z == 1) - rate(z == 0))),
        "TPR gap":   float(abs(rate((z == 1) & (y == 1)) - rate((z == 0) & (y == 1)))),
        "FPR gap":   float(abs(rate((z == 1) & (y == 0)) - rate((z == 0) & (y == 0)))),
        "P(yhat=1|z=1)": float(rate(z == 1)),
        "P(yhat=1|z=0)": float(rate(z == 0)),
    }


def evaluate_probs(p, y, z, thr_m=0.5, thr_f=0.5):
    """Metrics from predicted probabilities with (optionally) per-group thresholds."""
    p, y, z = np.asarray(p, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    yhat = np.where(z == 1, p >= thr_m, p >= thr_f).astype(float)
    out = _gaps(yhat, y, z)
    out["thr_M"], out["thr_F"] = float(thr_m), float(thr_f)
    return out


@torch.no_grad()
def predict_proba(predictor, X):
    predictor.eval()
    return torch.sigmoid(predictor(X)).cpu().numpy()


@torch.no_grad()
def evaluate(predictor, X, y, z, threshold=0.5):
    return _gaps((predict_proba(predictor, X) >= threshold).astype(float),
                 y.numpy(), z.numpy())


@torch.no_grad()
def evaluate_thresholds(predictor, X, y, z, thr_m, thr_f):
    return evaluate_probs(predict_proba(predictor, X), y.numpy(), z.numpy(), thr_m, thr_f)
