"""Nested random-feature fits and zero-mean loss integration controls."""

import numpy as np
import torch
from .compute import tt, cpu
from .models import arccos
from .io import data, dump, save_table, sha_array
from .config import PROTOCOL, RESULTS


def inverse(ev, penalties):
    penalties = np.asarray(penalties)
    finite = np.where(np.isfinite(penalties), penalties, 1.0)
    w = 1 / (ev[:, None] + 0.5 * tt(finite)[None, :]).clamp_min(1e-300)
    for j in np.flatnonzero(penalties == 0):
        cutoff = 1e-12 * ev.max()
        w[:, j] = torch.where(ev > cutoff, 1 / ev.clamp_min(cutoff), 0.0)
    w[:, np.isinf(penalties)] = 0.0
    return w


def fit_surface(group, draws=128, widths=None, penalties=None):
    spec = PROTOCOL["surface"][group]
    tasks = spec["tasks"]
    widths = np.asarray(spec["P"] if widths is None else widths, dtype=int)
    penalties = np.asarray(spec["lambda_value"] if penalties is None else penalties, dtype=float)
    zs = [data(t) for t in tasks]
    X, O = zs[0]["X"], zs[0]["Xo"]
    n, no, nl, nu = len(X), len(O), len(zs), len(penalties)
    y = tt(np.column_stack([z["y"] for z in zs]))
    yo = tt(np.column_stack([z["yo"] for z in zs]))
    mean = y.mean(0)
    centered = y - mean
    G, C, Xt = tt(arccos(X, X)), tt(arccos(O, X)), tt(np.r_[X, O])
    ev, V = torch.linalg.eigh(G)
    wi = inverse(ev.clamp_min(0), penalties)
    alpha = (V @ (wi[:, :, None] * (V.T @ centered)[:, None, :]).reshape(n, -1)).reshape(n, nu, nl)
    kernel = mean + (C @ alpha.reshape(n, -1)).reshape(no, nu, nl)
    residual = kernel - yo[:, None, :]
    aa = (
        V
        @ ((V.T @ (C.T @ residual.reshape(no, -1))).reshape(n, nu, nl) * wi[:, :, None]).reshape(
            n, -1
        )
    ).reshape(n, nu, nl)
    mask = tt(
        (np.isfinite(penalties) & (penalties > 0) & (penalties >= spec["control_min"])).astype(
            float
        )
    )
    losses, controlled, trains, ranks = [], [], [], []
    for seed in range(draws):
        rng = np.random.default_rng(spec["seed"] + seed)
        A, B = torch.zeros_like(G), torch.zeros_like(C)
        done = 0
        lr, lc, lt, rk = [], [], [], []
        for P in widths:
            while done < P:
                count = min(4096, int(P) - done)
                phi = torch.relu(Xt @ tt(rng.normal(size=(count, X.shape[1])).T))
                A += phi[:n] @ phi[:n].T
                B += phi[n:] @ phi[:n].T
                done += count
            gp, cp = A / P, B / P
            values, vectors = torch.linalg.eigh((gp + gp.T) / 2)
            values = values.clamp_min(0)
            inv = inverse(values, penalties)
            rhs = vectors.T @ centered
            coef = (inv[:, :, None] * rhs[:, None, :]).reshape(n, -1)
            pred = mean + ((cp @ vectors) @ coef).reshape(no, nu, nl)
            train = mean + (vectors @ (values[:, None] * coef)).reshape(n, nu, nl)
            loss = (pred - yo[:, None, :]).square().mean(0)
            dc = ((cp - C) @ alpha.reshape(n, -1)).reshape(no, nu, nl)
            dg = ((gp - G) @ alpha.reshape(n, -1)).reshape(n, nu, nl)
            control = 2 * ((residual * dc).sum(0) - (aa * dg).sum(0)) / no
            control *= mask[:, None]
            if P < 18:
                control *= 0
            lr.append(cpu(loss))
            lc.append(cpu(loss - control))
            lt.append(cpu((train - y[:, None, :]).square().mean(0)))
            rk.append(int((values > 1e-12 * values.max()).sum()))
        losses.append(lr)
        controlled.append(lc)
        trains.append(lt)
        ranks.append(rk)
        if (seed + 1) % 16 == 0:
            print(group, seed + 1, "/", draws, flush=True)
    out = dict(
        loss=np.array(losses),
        controlled=np.array(controlled),
        train=np.array(trains),
        rank=np.array(ranks),
        P=widths,
        lambda_value=penalties,
        tasks=np.array(tasks),
        D=cpu((yo - mean).square().mean(0)),
        kernel_loss=cpu(residual.square().mean(0)),
        seed_base=spec["seed"],
    )
    folder = RESULTS / "arrays"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"surface_{group}.npz", **out)
    dump(
        folder / f"surface_{group}.json",
        dict(
            draws=draws,
            seed=spec["seed"],
            P=widths.tolist(),
            lambda_value=penalties.tolist(),
            input_hashes={k: sha_array(zs[0][k]) for k in ["X", "Xo"]},
        ),
    )
    export_surface(out)
    return out


def export_surface(a):
    rows = []
    for ti, task in enumerate(a["tasks"]):
        n = len(data(task)["X"])
        D = float(a["D"][ti])
        for pi, P in enumerate(a["P"]):
            raw = a["loss"][:, pi, :, ti]
            adj = a["controlled"][:, pi, :, ti]
            for li, lam in enumerate(a["lambda_value"]):
                mse = raw[:, li].mean()
                se = raw[:, li].std(ddof=1) / np.sqrt(len(raw))
                rows.append(
                    dict(
                        task=task,
                        P=int(P),
                        c=P / n,
                        lambda_value=lam,
                        maps=len(raw),
                        MSE=mse,
                        MSE_MCSE=se,
                        NMSE=mse / D,
                        NMSE_MCSE=se / D,
                        R2=1 - mse / D,
                        controlled_MSE=adj[:, li].mean(),
                        controlled_MCSE=adj[:, li].std(ddof=1) / np.sqrt(len(raw)),
                        train_MSE=a["train"][:, pi, li, ti].mean(),
                        rank_min=a["rank"][:, pi].min(),
                    )
                )
    save_table(f"surface_{a['tasks'][0]}.csv", rows, folder="surface_tables")
