"""Simulate the spherical linear-teacher model of Mei and Montanari."""

from __future__ import annotations

import json
import time
from .config import RESULTS

import numpy as np
import pandas as pd
import torch

P_GRID = np.array(
    [
        15,
        30,
        60,
        90,
        150,
        210,
        255,
        285,
        297,
        300,
        303,
        315,
        345,
        450,
        600,
        900,
        1500,
        3000,
        6000,
        12000,
        24000,
        32768,
    ]
)
LAMBDAS = np.r_[0.0, np.geomspace(1e-7, 1e6, 183), np.inf]
DATA_SEED = 20261115
MAP_SEED = 810000
M = 0.5


def sphere(rng, rows, dimension, radius=1.0):
    x = rng.standard_normal((rows, dimension))
    return radius * x / np.linalg.norm(x, axis=1, keepdims=True)


def kernel(x, z):
    cosine = np.clip(x @ z.T, -1.0, 1.0)
    return (np.sqrt(np.maximum(0.0, 1 - cosine**2)) + (np.pi - np.arccos(cosine)) * cosine) / (
        2 * np.pi
    )


def generate(dimension, snrs):
    rng = np.random.default_rng(DATA_SEED)
    x = sphere(rng, 300, dimension)
    o = sphere(rng, 4096, dimension)
    f, fo = np.sqrt(dimension) * x[:, 0], np.sqrt(dimension) * o[:, 0]
    eps, epso = rng.standard_normal(len(x)), rng.standard_normal(len(o))
    y = np.stack([f + eps / np.sqrt(s) for s in snrs], axis=1)
    yo = np.stack([fo + epso / np.sqrt(s) for s in snrs], axis=1)
    return x, o, y, yo, f, fo


def losses(g, c, y, yo, fo):
    n, cases = y.shape
    mean = y.mean(0)
    ev, v = torch.linalg.eigh((g + g.T) / 2)
    ev = ev.clamp_min(0.0)
    cutoff = 1e-12 * ev[-1]
    lam = torch.as_tensor(LAMBDAS, device=g.device)
    inv = 1 / (ev[:, None] + M * lam[None, :].clamp_min(1e-300))
    inv[:, 0] = torch.where(ev > cutoff, 1 / ev.clamp_min(cutoff), 0.0)
    inv[:, -1] = 0.0
    h = c @ v
    weights = (v.T @ (y - mean))[:, :, None] * inv[:, None, :]
    pred = (h @ weights.reshape(n, -1)).reshape(len(yo), cases, -1) + mean[None, :, None]
    mse = ((yo[:, :, None] - pred) ** 2).mean(0)
    signal_mse = ((fo[:, None, None] - pred) ** 2).mean(0)
    train_pred = (v @ (ev[:, None, None] * weights).reshape(n, -1)).reshape(n, cases, -1)
    train_mse = ((y[:, :, None] - mean[None, :, None] - train_pred) ** 2).mean(0)
    return (
        mse.cpu().numpy(),
        signal_mse.cpu().numpy(),
        train_mse.cpu().numpy(),
        int((ev > cutoff).sum()),
    )


def run(draws=128, widths=None, penalties=None):
    global P_GRID, LAMBDAS
    if widths is not None:
        P_GRID = np.asarray(widths)
    if penalties is not None:
        LAMBDAS = np.asarray(penalties)
    d, snrs, case_names = 30, [50.0], ["snr50"]
    out = RESULTS / "synthetic"
    cache = RESULTS / "arrays"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    started = time.time()
    x, o, y, yo, f, fo = generate(d, snrs)
    td = lambda a: torch.as_tensor(a, device=device)
    tx, to, ty, tyo, tfo = map(td, [x, o, y, yo, fo])
    denom = np.mean((yo - y.mean(0)) ** 2, axis=0)
    k_mse, k_signal, k_train, k_rank = losses(td(kernel(x, x)), td(kernel(o, x)), ty, tyo, tfo)
    shape = (draws, len(P_GRID), len(snrs), len(LAMBDAS))
    mse = np.empty(shape)
    signal_mse, train_mse = np.empty(shape), np.empty(shape)
    ranks = np.empty((draws, len(P_GRID)), dtype=int)
    for seed in range(draws):
        rng = np.random.default_rng(MAP_SEED + seed)
        w = td(sphere(rng, int(P_GRID[-1]), d, np.sqrt(d)))
        a, b = torch.relu(tx @ w.T), torch.relu(to @ w.T)
        g = torch.zeros((len(x), len(x)), device=device)
        c = torch.zeros((len(o), len(x)), device=device)
        previous = 0
        for j, p in enumerate(P_GRID):
            sl = slice(previous, int(p))
            g += a[:, sl] @ a[:, sl].T
            c += b[:, sl] @ a[:, sl].T
            vals = losses(g / p, c / p, ty, tyo, tfo)
            mse[seed, j], signal_mse[seed, j], train_mse[seed, j], ranks[seed, j] = vals
            previous = int(p)
        if (seed + 1) % 8 == 0 or seed + 1 == draws:
            print(f"synthetic: {seed + 1}/{draws}, {time.time() - started:.1f}s", flush=True)
    records, seed_records = [], []
    for k, name in enumerate(case_names):
        for j, p in enumerate(P_GRID):
            vals = mse[:, j, k]
            avg = vals.mean(0)
            se = vals.std(0, ddof=1) / np.sqrt(draws) if draws > 1 else avg * np.nan
            signal = signal_mse[:, j, k].mean(0)
            train = train_mse[:, j, k].mean(0)
            for l, lam in enumerate(LAMBDAS):
                records.append(
                    dict(
                        case=name,
                        P=p,
                        P_over_n=p / len(x),
                        penalty=lam,
                        mse=avg[l],
                        mse_mcse=se[l],
                        nmse=avg[l] / denom[k],
                        nmse_mcse=se[l] / denom[k],
                        signal_mse=signal[l],
                        train_mse=train[l],
                        mean_rank=ranks[:, j].mean(),
                        benchmark_mse=denom[k],
                        maps=draws,
                    )
                )
            best = int(avg.argmin())
            for seed in range(draws):
                seed_records.append(
                    dict(
                        case=name,
                        map_seed=MAP_SEED + seed,
                        P=p,
                        ridgeless_nmse=vals[seed, 0] / denom[k],
                        mean_optimal_penalty=LAMBDAS[best],
                        nmse_at_mean_optimum=vals[seed, best] / denom[k],
                        ridgeless_train_mse=train_mse[seed, j, k, 0],
                        rank=ranks[seed, j],
                    )
                )
        for l, lam in enumerate(LAMBDAS):
            records.append(
                dict(
                    case=name,
                    P=np.inf,
                    P_over_n=np.inf,
                    penalty=lam,
                    mse=k_mse[k, l],
                    mse_mcse=0.0,
                    nmse=k_mse[k, l] / denom[k],
                    nmse_mcse=0.0,
                    signal_mse=k_signal[k, l],
                    train_mse=k_train[k, l],
                    mean_rank=k_rank,
                    benchmark_mse=denom[k],
                    maps=0,
                )
            )
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        RESULTS / "data/synthetic_surface.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(seed_records).to_csv(out / "mapping_diagnostics.csv", index=False)
    np.savez_compressed(
        cache / "synthetic_mapping_losses.npz",
        mse=mse,
        signal_mse=signal_mse,
        train_mse=train_mse,
        ranks=ranks,
        P=P_GRID,
        penalty=LAMBDAS,
        x=x,
        xo=o,
        y=y,
        yo=yo,
        f=f,
        fo=fo,
    )
    metadata = dict(
        paper="Mei and Montanari, arXiv:1908.05355v5",
        paper_url="https://arxiv.org/html/1908.05355v5",
        protocol="synthetic",
        n=len(x),
        d=d,
        n_test=len(o),
        snrs=[s if np.isfinite(s) else "infinity" for s in snrs],
        cases=case_names,
        data_seed=DATA_SEED,
        mapping_seed_start=MAP_SEED,
        maps=draws,
        m=M,
        P=P_GRID.tolist(),
        penalty=[v if np.isfinite(v) else "infinity" for v in LAMBDAS],
        input="Uniform unit sphere; paper input is sqrt(d) times this input",
        weights="Independent uniform sphere of radius sqrt(d)",
        teacher="sqrt(d) * x[0]; population signal variance = 1",
        labels="Add independent Gaussian noise with variance 1/SNR",
        mean_handling="Subtract training label mean; add it back to each forecast",
        penalty_conversion="lambda_paper = (d/n) * m * lambda_here",
        aggregation="Mean single-model MSE across mappings, not MSE of an ensemble",
        uncertainty="Mapping MCSE conditional on one fixed train/test sample",
        pseudoinverse_relative_cutoff=1e-12,
        seconds=time.time() - started,
    )
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(protocol="synthetic", seconds=metadata["seconds"])), flush=True)
