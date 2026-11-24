
import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import set_seed, SEED
from models import Predictor, Adversary, GradReverse
from metrics import evaluate


def _adversary_loss(A, adv_in, yb, zb, mode):
    """BCE for predicting z, restricted to y=1 examples when mode == 'eq_opp'."""
    z_logit = A(adv_in, yb)
    if mode == "eq_opp":
        mask = yb > 0.5
        if not bool(mask.any()):
            return z_logit.sum() * 0.0
        return F.binary_cross_entropy_with_logits(z_logit[mask], zb[mask])
    return F.binary_cross_entropy_with_logits(z_logit, zb)


def train(data, mode="eq_odds", alpha=1.0, debias=True, epochs=20, batch_size=128,
          lr_p=1e-3, lr_a=1e-2, seed=SEED, verbose=True,
          adv_hidden=None, adv_input="logit", alpha_schedule="const",
          use_grl=False, tag=""):
    """Train the predictor (and adversary) on the Adult training split.

    Parameters
    ----------
    mode          : "parity" | "eq_odds" | "eq_opp"
    debias        : False for the plain baseline
    adv_hidden    : hidden units inside the adversary (None = single linear)
    adv_input     : "logit" (paper) or "hidden" (Q11: 64-d last hidden layer)
    alpha_schedule: "const" or "sqrt" (Q7: alpha_t = alpha * sqrt(t / T))
    use_grl       : Q10 gradient-reversal path, one ordinary loss.backward()
    """
    assert mode in ("parity", "eq_odds", "eq_opp")
    assert adv_input in ("logit", "hidden")
    assert alpha_schedule in ("const", "sqrt")
    if use_grl:
        assert adv_input == "logit", "gradient reversal is implemented for the logit input"

    set_seed(seed)
    device = data["device"]
    P = Predictor(data["n_features"]).to(device)
    A = Adversary(mode=mode, hidden=adv_hidden, input_kind=adv_input).to(device) if debias else None

    opt_p = torch.optim.Adam(P.parameters(), lr=lr_p)
    opt_a = torch.optim.Adam(A.parameters(), lr=lr_a) if debias else None
    sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=epochs)  # decaying LR helps stability
    loader = DataLoader(data["train_ds"], batch_size=batch_size, shuffle=True)
    p_params = list(P.parameters())
    total_steps = epochs * len(loader)
    history = []
    prefix = f"[{tag}] " if tag else ""

    for epoch in range(1, epochs + 1):
        P.train()
        if A is not None:
            A.train()
        tot_lp = tot_la = 0.0
        alpha_epoch = alpha

        for step, (xb, yb, zb) in enumerate(loader, start=1):
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            global_step = (epoch - 1) * len(loader) + step
            alpha_t = alpha * math.sqrt(global_step / total_steps) if alpha_schedule == "sqrt" else alpha
            alpha_epoch = alpha_t

            # ---- 1. Forward pass ----
            if adv_input == "hidden":
                y_logit, adv_in = P(xb, return_hidden=True)
            else:
                y_logit = P(xb)
                adv_in = y_logit
            loss_p = F.binary_cross_entropy_with_logits(y_logit, yb)

            # ---- 2. Predictor update ----
            if not debias:
                opt_p.zero_grad()
                loss_p.backward()
                opt_p.step()
                loss_a = torch.tensor(0.0)
            elif use_grl:
                # Q10: one ordinary backward; the GRL flips the adversary's gradient
                opt_p.zero_grad()
                opt_a.zero_grad()
                loss_a = _adversary_loss(A, GradReverse.apply(adv_in, alpha_t), yb, zb, mode)
                (loss_p + loss_a).backward()
                opt_p.step()
                opt_a.step()
            else:
                loss_a = _adversary_loss(A, adv_in, yb, zb, mode)
                opt_p.zero_grad()
                grad_p = torch.autograd.grad(loss_p, p_params, retain_graph=True, allow_unused=True)
                grad_a = torch.autograd.grad(loss_a, p_params, retain_graph=True, allow_unused=True)
                for w, gp, ga in zip(p_params, grad_p, grad_a):
                    if gp is None:
                        gp = torch.zeros_like(w)
                    if ga is None:
                        ga = torch.zeros_like(w)
                    unit_ga = ga / (ga.norm() + 1e-8)
                    proj = (gp * unit_ga).sum() * unit_ga   # part of gp that helps the adversary
                    w.grad = gp - proj - alpha_t * ga       # the paper's update rule
                opt_p.step()

            # ---- 3. Adversary update (on detached predictor output) ----
            if debias and not use_grl:
                opt_a.zero_grad()
                _adversary_loss(A, adv_in.detach(), yb, zb, mode).backward()
                opt_a.step()

            tot_lp += loss_p.item() * len(xb)
            tot_la += loss_a.item() * len(xb)

        sched_p.step()
        n = len(data["train_ds"])
        m = evaluate(P, data["X_test"], data["y_test"], data["z_test"])
        history.append({"epoch": epoch, "loss_P": tot_lp / n, "loss_A": tot_la / n,
                        "alpha": alpha_epoch, **m})
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"{prefix}epoch {epoch:3d} | L_P {tot_lp/n:.4f} | L_A {tot_la/n:.4f} | "
                  f"acc {m['accuracy']:.4f} | DP {m['DP gap']:.3f} | "
                  f"TPR {m['TPR gap']:.3f} | FPR {m['FPR gap']:.3f}")
    return P, pd.DataFrame(history)
