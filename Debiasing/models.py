
import torch
import torch.nn as nn


class Predictor(nn.Module):
    """MLP with two 64-unit hidden layers; the last hidden layer is exposed for Q11."""

    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_hidden=False):
        h = self.body(x)
        logit = self.head(h).squeeze(-1)            # logit for y
        if return_hidden:
            return logit, h                         # 64-d representation (Q11)
        return logit


class Adversary(nn.Module):
    """Adversary that predicts z from the predictor's output.

    mode="parity"   : sees sigmoid(c * y_logit)
    mode="eq_odds"  : sees [s, s*y, s*(1-y), y]  (the paper's equalized-odds model)
    mode="eq_opp"   : sees s, but the *loss* is restricted to y = 1 (Q9)

    input_kind="hidden" feeds the predictor's 64-d last hidden layer instead of
    the output (Q11); in that case ``c`` is unused and the features are the
    representation itself.

    ``hidden`` adds a ReLU hidden layer to the adversary itself (Q6).
    """

    def __init__(self, mode="eq_odds", n_in=None, hidden=None, input_kind="logit"):
        super().__init__()
        assert mode in ("parity", "eq_odds", "eq_opp")
        assert input_kind in ("logit", "hidden")
        assert not (input_kind == "hidden" and mode != "parity"), \
            "hidden-input adversary is parity-style (no y feature)"
        self.mode = mode
        self.input_kind = input_kind
        self.c = nn.Parameter(torch.ones(1))

        if n_in is None:
            if input_kind == "hidden":
                n_in = 64
            elif mode == "eq_odds":
                n_in = 4
            else:
                n_in = 1
        self.n_in = n_in

        if hidden:
            self.lin = nn.Sequential(
                nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.lin = nn.Linear(n_in, 1)

    def features(self, y_logit, y_true=None):
        if self.input_kind == "hidden":
            return y_logit                        # already a 64-d representation
        s = torch.sigmoid((1 + self.c.abs()) * y_logit)
        if self.mode == "eq_odds":
            return torch.stack([s, s * y_true, s * (1 - y_true), y_true], dim=-1)
        return s.unsqueeze(-1)                    # parity and eq_opp

    def forward(self, y_logit, y_true=None):
        return self.lin(self.features(y_logit, y_true)).squeeze(-1)   # logit for z


class GradReverse(torch.autograd.Function):
    """Identity forward, gradient multiplied by -alpha backward (Q10)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
