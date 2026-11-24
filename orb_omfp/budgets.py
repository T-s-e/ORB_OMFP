"""Finite-feature budgets with independent draws and penalty retuning."""

import time
import hashlib
import numpy as np
import pandas as pd
import torch
from .config import RESULTS, TASKS, GROUPS, PROTOCOL, anchor
from .io import data, dump, sha
from .models import Model
from .compute import tt, cpu

OUT = RESULTS / "budget"
CACHE = RESULTS / "arrays/budget"
SEED = PROTOCOL["budget_seed"]


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.17g")


def dump_json(name, obj):
    dump(OUT / name, obj)


def setup(task):
    z = data(task)
    model = Model(z["X"], z["Xo"], z["y"])
    lam = float(anchor(task)["lambda_star"])
    kernel = model.predict([lam])[0] - model.mean
    e = z["yo"] - model.mean
    beta = e @ kernel / (kernel @ kernel)
    r2 = (2 * e @ kernel - kernel @ kernel) / (e @ e)
    return z, model, lam, e, beta, r2


def proposals():
    rows = []
    for task in TASKS:
        z, model, lam, e, beta, r2 = setup(task)
        kappa = float(anchor(task)["kappa"])
        for mult in [2, 4, 9, 19, 40, 99, 250]:
            P = int(np.ceil(kappa * mult))
            rows.append(
                dict(
                    task=task,
                    kind="nominal",
                    mult=mult,
                    q=mult / (mult + 1),
                    P=P,
                    kappa=kappa,
                    beta=beta,
                    lambda_kernel=lam,
                    R2_kernel=r2,
                )
            )
        for q in [0.95, 0.99]:
            P = int(np.ceil(kappa * q / ((1 - q) * (2 * beta - 1))))
            rows.append(
                dict(
                    task=task,
                    kind="calibration_diagnostic",
                    mult=np.nan,
                    q=q,
                    P=P,
                    kappa=kappa,
                    beta=beta,
                    lambda_kernel=lam,
                    R2_kernel=r2,
                )
            )
        for P in [32768, 65536, 200000]:
            rows.append(
                dict(
                    task=task,
                    kind="large_P",
                    mult=np.nan,
                    q=np.nan,
                    P=P,
                    kappa=kappa,
                    beta=beta,
                    lambda_kernel=lam,
                    R2_kernel=r2,
                )
            )
    write("frozen_cases.csv", rows)
    return pd.DataFrame(rows)


def fit(group, draws=128):
    started = time.time()
    table = pd.read_csv(OUT / "frozen_cases.csv")
    tasks = GROUPS[group]
    widths = np.sort(table[table.task.isin(tasks)].P.unique())
    z = data(tasks[0])
    X, O = z["X"], z["Xo"]
    n, no = len(X), len(O)
    model = Model(X, O, z["y"])
    G, C = tt(model.G), tt(model.C)
    Xt = tt(np.r_[X, O])
    contexts = []
    ratios = np.unique(np.r_[2.0 ** (np.arange(-24, 49) / 8), 2.0 ** (np.arange(-96, 97) / 96)])
    for task in tasks:
        row = table[table.task == task].iloc[0]
        lam = row.lambda_kernel * ratios
        target = data(task)
        q = tt(target["y"] - target["y"].mean())
        e = tt(target["yo"] - target["y"].mean())
        inv = 1 / (tt(model.ev)[:, None] + tt(0.5 * lam)[None, :])
        alpha = tt(model.V) @ (inv * (tt(model.V).T @ q)[:, None])
        p0 = C @ alpha
        residual = e[:, None] - p0
        # Contract the first-order loss control before fitting random maps.
        kt_residual = tt(model.V) @ (inv * (tt(model.CV).T @ residual))
        contexts.append(
            dict(
                task=task,
                lam=lam,
                alpha=alpha,
                p0=p0,
                residual=residual,
                kt_residual=kt_residual,
                q=q,
                e=e,
                ee=float(e @ e),
                j0=int(np.argmin(abs(ratios - 1))),
                records=[],
            )
        )
    predictions = {t: [] for t in tasks}
    for si in range(draws):
        rng = np.random.default_rng(SEED + si)
        A, B, done = torch.zeros_like(G), torch.zeros_like(C), 0
        per_task = {t: [] for t in tasks}
        loss_task = {t: [] for t in tasks}
        for P in widths:
            while done < P:
                count = min(4096, int(P) - done)
                phi = torch.relu(Xt @ tt(rng.normal(size=(count, X.shape[1])).T))
                A += phi[:n] @ phi[:n].T
                B += phi[n:] @ phi[:n].T
                done += count
            gp, cp = A / P, B / P
            ev, V = torch.linalg.eigh((gp + gp.T) / 2)
            cv = cp @ V
            for c in contexts:
                inv = 1 / (ev.clamp_min(0)[:, None] + tt(0.5 * c["lam"])[None, :])
                pred = cv @ (inv * (V.T @ c["q"])[:, None])
                r2 = (2 * c["e"] @ pred - pred.square().sum(0)) / c["ee"]
                dg_alpha = (gp - G) @ c["alpha"]
                dc_alpha = (cp - C) @ c["alpha"]
                control = (
                    2
                    * ((c["residual"] * dc_alpha).sum(0) - (c["kt_residual"] * dg_alpha).sum(0))
                    / c["ee"]
                )
                loss_task[c["task"]].append(cpu(torch.stack([r2, control])))
                per_task[c["task"]].append(cpu(pred[:, c["j0"]]))
        for c in contexts:
            c["records"].append(loss_task[c["task"]])
            predictions[c["task"]].append(per_task[c["task"]])
        if (si + 1) % 16 == 0:
            print(group, si + 1, "/", draws, "seconds", round(time.time() - started, 1), flush=True)
    for c in contexts:
        np.savez_compressed(
            CACHE / f"{c['task']}.npz",
            P=widths,
            lambda_value=c["lam"],
            losses=np.asarray(c["records"]),
            prediction=np.asarray(predictions[c["task"]]),
            j0=c["j0"],
            kernel_prediction=cpu(c["p0"]),
            seed=SEED,
        )
    dump_json(
        f"fit_{group}.json",
        dict(
            group=group,
            maps=draws,
            seed=SEED,
            P=widths.tolist(),
            seconds=time.time() - started,
            source_sha256=sha(__file__),
            data_hashes={k: digest_array(v) for k, v in [("X", X), ("Xo", O)]},
            batch_features=4096,
            query_labels_used_to_fit_weights=False,
            lambda_selection="historical test oracle, averaged single-map controlled loss",
        ),
    )


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def crossfit(a):
    even = np.arange(len(a)) % 2 == 0
    ja = int(a[even].mean(0).argmax())
    jb = int(a[~even].mean(0).argmax())
    va, vb = a[~even, ja], a[even, jb]
    value = 0.5 * (va.mean() + vb.mean())
    se = 0.5 * np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
    return value, se, ja, jb


def boot_crossfit(a, count=2000):
    rng = np.random.default_rng(22115400)
    x, y = a[::2], a[1::2]
    vals = []
    for first in range(0, count, 100):
        ia = rng.integers(len(x), size=(min(100, count - first), len(x)))
        ib = rng.integers(len(y), size=(len(ia), len(y)))
        ma, mb = x[ia].mean(1), y[ib].mean(1)
        ja, jb = ma.argmax(1), mb.argmax(1)
        ix = np.arange(len(ia))
        vals.extend(0.5 * (mb[ix, ja] + ma[ix, jb]))
    return np.std(vals, ddof=1)


def reliability(pred):
    s = len(pred)
    sq = np.sum(pred * pred, axis=1)
    total = pred.sum(0)
    shared = (total @ total - sq.sum()) / (s * (s - 1))
    return shared / sq.mean()


def reliability_se(pred):
    loo = np.array([reliability(np.delete(pred, i, axis=0)) for i in range(len(pred))])
    return np.sqrt((len(pred) - 1) / len(pred) * np.sum((loo - loo.mean()) ** 2))


def summaries():
    cases = pd.read_csv(OUT / "frozen_cases.csv")
    rows, curves = [], []
    for task in TASKS:
        z, model, lam, e, beta, r2 = setup(task)
        if not (CACHE / f"{task}.npz").exists():
            continue
        with np.load(CACHE / f"{task}.npz") as f:
            a = dict(f)
        draws = a["losses"].shape[0]
        for pi, P in enumerate(a["P"]):
            raw, cv = a["losses"][:, pi, 0], a["losses"][:, pi, 1]
            corrected = raw - cv
            value, se, ja, jb = crossfit(corrected)
            bse = boot_crossfit(corrected)
            j = int(corrected.mean(0).argmax())
            j0 = int(a["j0"])
            ref = cases[cases.task == task].iloc[0]
            pred = a["prediction"][:, pi]
            rel = reliability(pred)
            rel_se = reliability_se(pred)
            rows.append(
                dict(
                    task=task,
                    P=int(P),
                    maps=draws,
                    kappa=ref.kappa,
                    lambda_kernel=lam,
                    R2_kernel=r2,
                    beta=beta,
                    gamma=2 * beta - 1,
                    R2_tuned=value,
                    eta_tuned=value / r2,
                    eta_tuned_SE=se / r2,
                    eta_bootstrap_SE=bse / r2,
                    eta_grid_max=corrected.mean(0)[j] / r2,
                    lambda_P=a["lambda_value"][j],
                    lambda_ratio=a["lambda_value"][j] / lam,
                    lambda_half_A=a["lambda_value"][ja],
                    lambda_half_B=a["lambda_value"][jb],
                    eta_fixed=corrected[:, j0].mean() / r2,
                    eta_fixed_SE=corrected[:, j0].std(ddof=1) / np.sqrt(draws) / r2,
                    eta_raw_fixed=raw[:, j0].mean() / r2,
                    eta_raw_fixed_SE=raw[:, j0].std(ddof=1) / np.sqrt(draws) / r2,
                    control_mean_Z=cv[:, j0].mean() / (cv[:, j0].std(ddof=1) / np.sqrt(draws)),
                    reliability=rel,
                    reliability_SE=rel_se,
                    nominal=P / (P + ref.kappa),
                    calibrated=P / (P + ref.kappa / (2 * beta - 1)),
                    at_grid_edge=j in [0, len(a["lambda_value"]) - 1],
                )
            )
            for li, l in enumerate(a["lambda_value"]):
                curves.append(
                    dict(
                        task=task,
                        P=int(P),
                        lambda_value=l,
                        R2=corrected[:, li].mean(),
                        R2_SE=corrected[:, li].std(ddof=1) / np.sqrt(draws),
                        R2_raw=raw[:, li].mean(),
                        R2_raw_SE=raw[:, li].std(ddof=1) / np.sqrt(draws),
                    )
                )
        print("Summarized", task, flush=True)
    write("budget_results.csv", rows)
    write("retuned_curves.csv", curves)
    a = pd.DataFrame(rows)
    budgets = cases.merge(a, on=["task", "P"], suffixes=("_frozen", ""), validate="many_to_one")
    budgets = budgets[(budgets.kind != "large_P") & budgets.q.isin([0.95, 0.99])]
    write("budget_table.csv", budgets)


def run(group=None, draws=128):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    proposals()
    for name in [group] if group else GROUPS:
        fit(name, draws)
    summaries()
    table = pd.read_csv(OUT / "budget_table.csv").query("kind == 'nominal'").copy()
    table["retained_percent"] = 100 * table.eta_tuned
    table["MCSE_percentage_points"] = 100 * table.eta_bootstrap_SE
    table["reliability_percent"] = 100 * table.reliability
    (RESULTS / "tables").mkdir(exist_ok=True, parents=True)
    table.to_csv(RESULTS / "tables/feature_budget.csv", index=False, float_format="%.17g")
    (RESULTS / "data").mkdir(exist_ok=True, parents=True)
    pd.read_csv(OUT / "budget_results.csv").to_csv(
        RESULTS / "data/budget_curves.csv", index=False, float_format="%.17g"
    )
