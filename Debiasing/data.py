
"""UCI Adult loader with a configurable protected attribute.

Two scenarios are supported:

    protected="sex"   ->  z = 1 if Male, 0 if Female   (sex is removed from X)
    protected="race"  ->  z = 1 if White, 0 otherwise  (race is removed from X, sex kept)

The loader tries OpenML first (the source the workshop uses), then the Kaggle
mirror (``kagglehub``), then the raw UCI CSV files.  That way a flaky OpenML
endpoint does not kill a Kaggle session.

The dictionary returned by :func:`load_adult` contains:

    train_ds    TensorDataset of (X_train, y_train, z_train)
    X_val / y_val / z_val    15% validation split (needed for Q12)
    X_test / y_test / z_test 15% held-out test split
    n_features, feature_names, device, protected
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

# Column names used by the OpenML/UCI versions of Adult.  Kaggle's mirror uses
# dots instead of dashes for some of them.
CANONICAL = {
    "income": "class",
    "education.num": "education-num",
    "marital.status": "marital-status",
    "capital.gain": "capital-gain",
    "capital.loss": "capital-loss",
    "hours.per.week": "hours-per-week",
    "native.country": "native-country",
}

UCI_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]
UCI_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _from_openml():
    from sklearn.datasets import fetch_openml
    return fetch_openml("adult", version=2, as_frame=True).frame


def _from_kagglehub():
    import glob
    import os
    import kagglehub

    path = kagglehub.dataset_download("uciml/adult-census-income")
    csv = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))[0]
    return pd.read_csv(csv, na_values=["?", " ?"])


def _from_uci():
    train = pd.read_csv(UCI_TRAIN, header=None, names=UCI_COLUMNS,
                        skipinitialspace=True, na_values=["?", " ?"])
    test = pd.read_csv(UCI_TEST, header=None, names=UCI_COLUMNS, skiprows=1,
                       skipinitialspace=True, na_values=["?", " ?"])
    return pd.concat([train, test], ignore_index=True)


def _load_frame(verbose=True):
    errors = []
    for name, loader in (("OpenML", _from_openml),
                         ("kagglehub", _from_kagglehub),
                         ("UCI CSV", _from_uci)):
        try:
            df = loader()
            if verbose:
                print(f"Loaded Adult from {name}: {df.shape}")
            return df
        except Exception as exc:      # noqa: BLE001 - fall back to the next source
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load the Adult dataset:\n  " + "\n  ".join(errors))


def _canonicalize(df):
    df = df.copy()
    df.columns = [CANONICAL.get(str(c).strip(), str(c).strip()) for c in df.columns]

    numeric = ["age", "fnlwgt", "education-num", "capital-gain",
               "capital-loss", "hours-per-week"]
    for c in df.columns:
        if c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()

    df = df.replace({"?": pd.NA, " ?": pd.NA, "": pd.NA})
    return df.dropna().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_adult(seed=SEED, device=None, verbose=True, protected="sex"):
    """Returns a dict with the train/val/test tensors and metadata.

    ``protected`` selects the binary attribute ``z`` and which column is
    removed from the features.  Splits are 70% / 15% / 15%, stratified on the
    four (y, z) combinations so both groups and both labels survive every split.
    """
    assert protected in ("sex", "race"), "protected must be 'sex' or 'race'"
    device = device or get_device()
    df = _canonicalize(_load_frame(verbose=verbose))

    cls = df["class"].astype(str).str.strip().str.rstrip(".")
    y = cls.str.startswith(">50K").astype(np.float32).values

    if protected == "sex":
        z = (df["sex"].astype(str).str.strip() == "Male").astype(np.float32).values
        drop = ["class", "sex", "fnlwgt"]
    else:
        z = (df["race"].astype(str).str.strip().str.lower() == "white").astype(np.float32).values
        drop = ["class", "race", "fnlwgt"]

    X_df = df.drop(columns=drop)
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    X_df = pd.get_dummies(X_df, columns=cat_cols, dtype=np.float32)
    feature_names = X_df.columns.tolist()

    X = X_df.values.astype(np.float32)
    labels = (y.astype(int) * 2 + z.astype(int))

    # 70 / 15 / 15, stratified on (y, z)
    X_train, X_rest, y_train, y_rest, z_train, z_rest = train_test_split(
        X, y, z, test_size=0.30, random_state=seed, stratify=labels)
    rest_labels = (y_rest.astype(int) * 2 + z_rest.astype(int))
    X_val, X_test, y_val, y_test, z_val, z_test = train_test_split(
        X_rest, y_rest, z_rest, test_size=0.50, random_state=seed, stratify=rest_labels)

    # Standardize numeric columns (fit on train only!)
    num_idx = [X_df.columns.get_loc(c) for c in num_cols]
    scaler = StandardScaler().fit(X_train[:, num_idx])
    X_train[:, num_idx] = scaler.transform(X_train[:, num_idx])
    X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])
    X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

    if verbose:
        print(f"Protected attribute: {protected}   (z=1 share {z.mean():.3f})")
        print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
        print(f"P(y=1) = {y.mean():.3f}   P(z=1) = {z.mean():.3f}")
        print(f"P(y=1 | z=1) = {y[z == 1].mean():.3f}   "
              f"P(y=1 | z=0) = {y[z == 0].mean():.3f}")

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        "train_ds": TensorDataset(t(X_train), t(y_train), t(z_train)),
        "X_val": t(X_val).to(device), "y_val": t(y_val), "z_val": t(z_val),
        "X_test": t(X_test).to(device), "y_test": t(y_test), "z_test": t(z_test),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "device": str(device),
        "protected": protected,
    }
