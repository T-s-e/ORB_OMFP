"""Forecast reliability and its single-feature variance coefficient."""

import numpy as np
import torch
from .config import GROUPS, PROTOCOL, RESULTS, anchor
from .io import data, save_table
from .models import Model
from .compute import tt, cpu


def coefficient(task, features=262144):
    z = data(task)
    model = Model(z["X"], z["Xo"], z["y"])
    lam = anchor(task)["lambda_star"]
    lm = 0.5 * lam
    Q = (model.V / (model.ev + lm)[None, :]) @ model.V.T
    K = model.C @ Q
    alpha = Q @ (z["y"] - z["y"].mean())
    prediction = model.C @ alpha
    shift = lm * K @ alpha
    monthly = task != "macro_treasury"
    batch = 8192 if monthly else 4096
    rng = np.random.default_rng(0 if monthly else 17173100)
    X, O, kt, at, st = tt(z["X"]), tt(z["Xo"]), tt(K), tt(alpha), tt(shift)
    values = []
    for first in range(0, features, batch):
        count = min(batch, features - first)
        w = (
            rng.normal(size=(X.shape[1], count))
            if monthly
            else rng.normal(size=(count, X.shape[1])).T
        )
        phi, psi = torch.relu(X @ tt(w)), torch.relu(O @ tt(w))
        ell = psi - kt @ phi
        score = at @ phi
        if monthly:
            value = (ell.square().sum(0) * score.square()).mean() - st @ st
        else:
            value = (ell * score[None, :] - st[:, None]).square().sum(0).mean()
        values.append(float(value) / (prediction @ prediction))
    return dict(
        task=task,
        lambda_value=lam,
        kappa=float(np.mean(values)),
        kappa_MCSE=float(np.std(values, ddof=1) / np.sqrt(len(values))),
        features=features,
        seed=0 if monthly else 17173100,
        batch=batch,
        test_labels_used=False,
    )


def statistic(pred):
    count = len(pred)
    square = np.sum(pred * pred, axis=1)
    total = pred.sum(0)
    shared = (total @ total - square.sum()) / (count * (count - 1))
    rho = (square.mean() - shared) / shared
    return float(rho)


def jackknife_rho(pred):
    blocks = np.array_split(np.arange(len(pred)), 8)
    loo = np.array([statistic(np.delete(pred, b, axis=0)) for b in blocks])
    return float(np.sqrt((len(blocks) - 1) / len(blocks) * np.sum((loo - loo.mean()) ** 2)))


def fit_group(group, draws=128, widths=None):
    tasks = GROUPS[group]
    zs = [data(t) for t in tasks]
    z = zs[0]
    X, O = z["X"], z["Xo"]
    n = len(X)
    widths = PROTOCOL["reliability"][group]["P"] if widths is None else widths
    model = Model(X, O, z["y"])
    G, C = tt(model.G), tt(model.C)
    Xt = tt(np.r_[X, O])
    contexts = []
    for task, a in zip(tasks, zs):
        contexts.append((anchor(task)["lambda_star"], tt(a["y"] - a["y"].mean())))
    records = []
    for seed in range(draws):
        rng = np.random.default_rng(PROTOCOL["reliability_seed"] + seed)
        A, B, done = torch.zeros_like(G), torch.zeros_like(C), 0
        predictions = []
        for P in widths:
            while done < P:
                count = min(4096, int(P) - done)
                phi = torch.relu(Xt @ tt(rng.normal(size=(count, X.shape[1])).T))
                A += phi[:n] @ phi[:n].T
                B += phi[n:] @ phi[:n].T
                done += count
            ev, V = torch.linalg.eigh((A + A.T) / (2 * P))
            cv = (B / P) @ V
            predictions.append(
                [cpu(cv @ ((V.T @ q) / (ev.clamp_min(0) + 0.5 * lam))) for lam, q in contexts]
            )
        records.append(predictions)
        if (seed + 1) % 16 == 0:
            print("Reliability", group, seed + 1, "/", draws, flush=True)
    records = np.asarray(records)
    folder = RESULTS / "arrays"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"reliability_{group}.npz",
        prediction=records,
        P=widths,
        tasks=tasks,
        seed=PROTOCOL["reliability_seed"],
    )
    rows = []
    for pi, P in enumerate(widths):
        for ti, task in enumerate(tasks):
            rho = statistic(records[:, pi, ti])
            se = jackknife_rho(records[:, pi, ti])
            rows.append(
                dict(
                    task=task,
                    P=P,
                    transformed=P * rho,
                    transformed_SE=P * se,
                    kappa=anchor(task)["kappa"],
                    maps=draws,
                )
            )
    save_table(f"reliability_{group}.csv", rows, folder="reliability_tables")
    return records


def run(group=None, draws=128, features=262144):
    groups = [group] if group else GROUPS
    tasks = [t for g in groups for t in GROUPS[g]]
    save_table("reliability_coefficients.csv", [coefficient(t, features) for t in tasks])
    for g in groups:
        fit_group(g, draws)
    import pandas as pd

    frames = [pd.read_csv(p) for p in sorted((RESULTS / "reliability_tables").glob("*.csv"))]
    save_table("reliability_transform.csv", pd.concat(frames), folder="data")
