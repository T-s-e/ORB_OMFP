"""The four training-only penalty selectors used in the paper."""

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize_scalar
from .config import U, RESULTS
from .data import sources, inputs, target
from .io import data, save_table
from .models import Model


def components(model):
    basis = helmert(model.n, full=False).T
    values, vectors = np.linalg.eigh(basis.T @ (model.G / model.m) @ basis)
    energy = ((basis @ vectors).T @ model.y) ** 2
    return np.maximum(values, 1e-12), energy


def moments(model):
    values, energy = components(model)
    centered = values - values.mean()
    vf = centered @ energy / (centered @ centered)
    sigma2 = energy.mean() - vf * values.mean()
    lam = sigma2 / vf if vf > 0 and sigma2 > 0 else np.inf
    return float(lam), float(sigma2), float(vf)


def reml(model):
    values, energy = components(model)

    def objective(lam):
        if np.isinf(lam):
            return len(values) * np.log(energy.mean())
        variance = (values + lam) / (values.mean() + lam)
        return np.log(variance).sum() + len(values) * np.log(np.mean(energy / variance))

    return choose(U, np.array([objective(v) for v in U]), objective)


def choose(grid, scores, objective):
    j = int(np.argmin(scores))
    if scores[-1] <= scores[j] + 1e-11:
        return np.inf
    if j == 0:
        return 0.0
    fit = minimize_scalar(
        lambda v: objective(np.exp(v)), bounds=np.log(grid[[j - 1, j + 1]]), method="bounded"
    )
    return float(np.exp(fit.x))


def splits(n, horizon, rolling):
    purge = 12 if horizon == 12 else 0
    if not rolling:
        yield np.arange(n - 96 - purge), np.arange(n - 96, n)
    else:
        for start in range(120, n, 12):
            yield np.arange(start - purge), np.arange(start, min(start + 12, n))


def select_history(source, task, rolling):
    n, horizon = len(source["train"]), source["horizon"]
    grid = U if horizon == 12 else np.r_[0.0, np.geomspace(1e-7, 1e8, 361), np.inf]
    models, counts, fold_rows = [], [], []
    for j, (tr, va) in enumerate(splits(n, horizon, rolling)):
        fold = dict(source, train=source["train"][tr], test=source["train"][va])
        X, O, _ = inputs(fold)
        y, yo, _ = target(fold)
        model = Model(X, O, y)
        models.append((model, yo))
        counts.append(len(va))
        fold_rows.append(
            dict(
                task=task,
                method="rolling_cv" if rolling else "validation",
                fold=j,
                fit_n=len(tr),
                validation_n=len(va),
                fit_start=str(fold["train"][0]),
                fit_end=str(fold["train"][-1]),
                validation_start=str(fold["test"][0]),
                validation_end=str(fold["test"][-1]),
                purge=12 if horizon == 12 else 0,
                active_post_validation_embargo=0,
                final_test_used=False,
            )
        )
    scores = np.average([m.loss(grid, yo) for m, yo in models], axis=0, weights=counts)

    def objective(lam):
        return sum(len(yo) * m.loss(lam, yo)[0] for m, yo in models) / sum(counts)

    return (
        choose(grid, scores, objective),
        fold_rows,
        [
            dict(
                task=task,
                method="rolling_cv" if rolling else "validation",
                lambda_value=v,
                validation_MSE=l,
            )
            for v, l in zip(grid, scores)
        ],
    )


def run():
    refs = pd.read_csv(RESULTS / "tables/kernel_summary.csv").set_index("task")
    rows, folds, curves, estimates = [], [], [], []
    for task, source in sources().items():
        z = data(task)
        model = Model(z["X"], z["Xo"], z["y"])
        moment, sigma2, vf = moments(model)
        choices = {"kernel_moments": moment, "reml": reml(model)}
        estimates.append(dict(task=task, sigma2=sigma2, vf=vf, lambda_hat=moment))
        for method, rolling in [("validation", False), ("rolling_cv", True)]:
            choices[method], fs, cs = select_history(source, task, rolling)
            folds.extend(fs)
            curves.extend(cs)
        ref = refs.loc[task]
        for method, lam in choices.items():
            r2 = float(1 - model.loss(lam, z["yo"])[0] / ref.D)
            rows.append(
                dict(
                    task=task,
                    method=method,
                    lambda_hat=lam,
                    lambda_star=ref.lambda_star,
                    lambda_ratio=lam / ref.lambda_star,
                    R2=r2,
                    R2_percent=100 * r2,
                    retained_gain=r2 / ref.R2_star,
                    no_signal=np.isinf(lam),
                    test_used_for_selection=False,
                    **{f"inside_{int(100 * q)}": r2 >= q * ref.R2_star for q in [0.9, 0.95, 0.99]},
                )
            )
        print("Selected penalties:", task, flush=True)
    for name, values in [
        ("selection.csv", rows),
        ("validation_folds.csv", folds),
        ("validation_curves_all_tasks.csv", curves),
        ("variance_components.csv", estimates),
    ]:
        save_table(name, values)
