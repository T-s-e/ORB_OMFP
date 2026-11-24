"""Kernel-limit curves, label sensitivity and near-optimal bands."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from .config import RESULTS, TASKS
from .io import data
from .models import Model

TABLES = RESULTS / "tables"
GRID = np.r_[0.0, np.geomspace(1e-7, 1e8, 601), np.inf]


def save(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / name, index=False, float_format="%.17g")


def optimum(model, yo):
    scores = model.loss(GRID, yo)
    j = scores.argmin()
    result = minimize_scalar(
        lambda x: model.loss(np.exp(x), yo)[0],
        bounds=np.log(GRID[[j - 1, j + 1]]),
        method="bounded",
        options={"xatol": 1e-11},
    )
    return float(np.exp(result.x))


def geometry(model, e, penalties):
    lam = np.atleast_1d(penalties)
    inv = model.inverse(lam)
    p = model.CV @ (inv * model.b[:, None])
    dp = -model.m * model.CV @ (inv**2 * model.b[:, None])
    norm2 = np.sum(p * p, axis=0)
    a = np.sqrt(norm2 / (e @ e))
    c = (e @ p) / np.sqrt((e @ e) * norm2)
    omega = -lam * np.sum(p * dp, axis=0) / norm2
    return (
        dict(a=a, cos=c, omega=omega, norm=np.sqrt(norm2), R2=2 * a * c - a * a, beta=c / a),
        p,
        dp,
    )


def delta(model):
    K = model.operator(0.0)
    H = K - K.mean(axis=1, keepdims=True)
    return float(np.sum(H * H) / model.no)


def main():
    profiles, summary, bands, path, band_geometry = [], [], [], [], []
    lengths, intervention, spectrum, checks, datasets = [], [], [], [], []
    length_summary, sensitivity, plateau, extremes = [], [], [], []
    for task in TASKS:
        z = data(task)
        X, O, y, yo = [z[k] for k in ["X", "Xo", "y", "yo"]]
        mod = Model(X, O, y)
        center = optimum(mod, yo)
        D = float(np.mean((yo - y.mean()) ** 2))
        r2 = float(1 - mod.loss(center, yo)[0] / D)
        e = yo - y.mean()
        gs, ps, _ = geometry(mod, e, center)
        omega, beta = float(gs["omega"][0]), float(gs["beta"][0])
        lam = np.unique(
            np.r_[np.geomspace(1e-7, 1e8, 601), np.geomspace(center / 20, center * 20, 601), center]
        )
        g, pp, dp = geometry(mod, e, lam)
        for j, v in enumerate(lam):
            path.append(
                dict(
                    task=task,
                    lambda_value=v,
                    lambda_ratio=v / center,
                    **{k: float(g[k][j]) for k in g},
                    t=g["norm"][j] / gs["norm"][0],
                    retained_gain=g["R2"][j] / r2,
                    power_t=(v / center) ** (-omega),
                )
            )
        for v, loss in zip(np.r_[0.0, lam, np.inf], mod.loss(np.r_[0.0, lam, np.inf], yo)):
            profiles.append(
                dict(task=task, lambda_value=v, MSE=loss, NMSE=loss / D, R2=1 - loss / D, D=D)
            )
        dl = delta(mod)
        Ks = mod.operator(center)
        hs = Ks - Ks.mean(axis=1, keepdims=True)
        dl_star = float(np.sum(hs * hs) / mod.no)
        summary.append(
            dict(
                task=task,
                n=len(y),
                no=len(yo),
                d=X.shape[1],
                D=D,
                lambda_star=center,
                R2_star=r2,
                R2_star_percent=100 * r2,
                omega_star=omega,
                beta_MZ=beta,
                gamma=2 * beta - 1,
                ridgeless_NMSE=mod.loss(0.0, yo)[0] / D,
                deltaL_0=dl,
                deltaL_star=dl_star,
                variance_reduction=dl / dl_star,
                neumann_threshold=mod.ev.max() / mod.m,
                threshold_over_star=mod.ev.max() / (mod.m * center),
                eigenvalues_above_star=int(np.sum(mod.ev > mod.m * center)),
            )
        )
        K = mod.operator(0.0)
        H = K - K.mean(1, keepdims=True)
        p0 = H @ y
        amp = float(p0 @ p0 / (e @ e))
        align = float(2 * e @ p0 / (e @ e))
        plateau.append(
            dict(
                task=task,
                NMSE=1 + amp - align,
                excess_NMSE=amp - align,
                squared_amplitude=amp,
                alignment=align,
                deltaL=dl,
                homoskedastic_weak_signal_NMSE=1 + dl,
                Gamma=(amp - align) / dl,
            )
        )
        R = np.sum(X * X, axis=1)
        Ro = np.sum(O * O, axis=1)
        qs = np.mean(K * K, axis=0)
        for i in range(len(y)):
            sensitivity.append(
                dict(
                    task=task,
                    month=str(z["train_ids"][i]),
                    radius2=R[i],
                    uncentered_sensitivity=qs[i],
                    direction_factor=qs[i] * R[i],
                )
            )
        length_summary.append(
            dict(
                task=task,
                mean_train_radius2=R.mean(),
                inverse_train_radius2_mean=np.mean(1 / R),
                length_factor=R.mean() * np.mean(1 / R),
                log_sensitivity_slope=np.polyfit(np.log(R), np.log(qs), 1)[0],
                core_norm=float(np.sum(K * K) / mod.no),
                centered_deltaL=dl,
                centering_subtraction=float(np.sum(K.mean(1) ** 2) * len(y) / mod.no),
                max_test_radius2=Ro.max(),
                max_radius_month=str(z["test_ids"][int(Ro.argmax())]),
            )
        )
        for j in np.argsort(p0 * p0)[-5:][::-1]:
            extremes.append(
                dict(
                    task=task,
                    month=str(z["test_ids"][j]),
                    radius2=Ro[j],
                    ridgeless_prediction_energy_share=p0[j] ** 2 / (p0 @ p0),
                )
            )
        for split, arr in [("train", X), ("test", O)]:
            ids = z["train_ids" if split == "train" else "test_ids"]
            datasets.append(
                dict(
                    task=task,
                    split=split,
                    n=len(arr),
                    d=arr.shape[1],
                    start=str(ids[0]),
                    end=str(ids[-1]),
                    mean_squared_length=np.mean(np.sum(arr * arr, axis=1)),
                )
            )
        for q in [0.90, 0.95, 0.99]:
            f = lambda v: 1 - mod.loss(v, yo)[0] / D - q * r2
            lo = brentq(f, center * 1e-6, center)
            hi = brentq(f, center, center * 1e6)
            s = np.sqrt(1 - q)
            plo, phi = center * (1 + s) ** (-1 / omega), center * (1 - s) ** (-1 / omega)
            norm = gs["norm"][0]
            nlo = brentq(
                lambda v: geometry(mod, e, v)[0]["norm"][0] / norm - (1 + s), center * 1e-6, center
            )
            nhi = brentq(
                lambda v: geometry(mod, e, v)[0]["norm"][0] / norm - (1 - s), center, center * 1e6
            )
            bands.append(
                dict(
                    task=task,
                    q=q,
                    lambda_star=center,
                    omega_star=omega,
                    empirical_lower=lo,
                    empirical_upper=hi,
                    predicted_lower=plo,
                    predicted_upper=phi,
                    exact_t_lower=nlo,
                    exact_t_upper=nhi,
                    max_edge_error=max(abs(plo / lo - 1), abs(phi / hi - 1)),
                    exact_t_max_edge_error=max(abs(nlo / lo - 1), abs(nhi / hi - 1)),
                )
            )
            geo, _, _ = geometry(mod, e, np.geomspace(lo, hi, 501))
            slope, intercept = np.polyfit(geo["a"], geo["cos"], 1)
            band_geometry.append(
                dict(
                    task=task,
                    q=q,
                    lower=lo,
                    upper=hi,
                    alignment_slope=slope,
                    alignment_intercept=intercept,
                    relative_alignment_residual=np.max(
                        abs(geo["cos"] - (intercept + slope * geo["a"]))
                    )
                    / abs(gs["cos"][0]),
                    collapse_max_error=np.max(
                        abs(geo["R2"] / r2 - (2 * geo["norm"] / norm - (geo["norm"] / norm) ** 2))
                    ),
                    omega_min=geo["omega"].min(),
                    omega_max=geo["omega"].max(),
                )
            )
        for tag, xx, oo in [
            ("original", X, O),
            ("train_unit", X / np.linalg.norm(X, axis=1)[:, None], O),
            ("test_unit", X, O / np.linalg.norm(O, axis=1)[:, None]),
            (
                "both_unit",
                X / np.linalg.norm(X, axis=1)[:, None],
                O / np.linalg.norm(O, axis=1)[:, None],
            ),
        ]:
            mm = mod if tag == "original" else Model(xx, oo, y)
            lengths.append(dict(task=task, intervention=tag, deltaL=delta(mm)))
        idx = int(np.argmin(np.linalg.norm(X, axis=1)))
        xx = X.copy()
        xx[idx] *= 2
        changed = Model(xx, O, y).operator(0.0)
        expected = mod.operator(0.0).copy()
        expected[:, idx] /= 2
        intervention.append(
            dict(
                task=task,
                row=idx,
                scale=2.0,
                relative_operator_error=np.linalg.norm(changed - expected)
                / np.linalg.norm(expected),
            )
        )

        # Check elasticity directly, including signed cross-direction weights.
        h = 1e-5
        np_, _, _ = geometry(mod, e, lam * np.exp(h))
        nm_, _, _ = geometry(mod, e, lam * np.exp(-h))
        fd = -(np.log(np_["norm"]) - np.log(nm_["norm"])) / (2 * h)
        inv = mod.inverse(center).ravel()
        contributions = mod.CV * (mod.b * inv)[None, :]
        p = ps.ravel()
        weights = (p @ contributions) / (p @ p)
        wmode = center / (center + mod.ev / mod.m)
        for i in np.argsort(mod.ev)[::-1]:
            spectrum.append(
                dict(
                    task=task,
                    mode=int(i),
                    tau=mod.ev[i] / mod.m,
                    signed_weight=weights[i],
                    shrinkage_elasticity=wmode[i],
                    omega_contribution=weights[i] * wmode[i],
                )
            )
        checks.append(
            dict(
                task=task,
                omega_derivative_error=float(np.max(abs(fd - g["omega"]))),
                omega_spectral_error=abs(weights @ wmode - omega),
                spectral_weight_sum_error=abs(weights.sum() - 1),
                full_path_norm_decreasing=bool(np.all(np.diff(g["norm"]) <= 1e-10)),
                identity_error=float(np.max(abs(g["R2"] - (1 - mod.loss(lam, yo) / D)))),
                negative_mode_weights=int(np.sum(weights < -1e-10)),
                signed_weight_min=float(weights.min()),
            )
        )
        assert checks[-1]["omega_derivative_error"] < 1e-7
        assert checks[-1]["omega_spectral_error"] < 1e-10
        print(task, "lambda*", center, "R2", r2, "omega", omega, flush=True)
    for name, rows in [
        ("kernel_summary.csv", summary),
        ("kernel_profiles.csv", profiles),
        ("kernel_bands.csv", bands),
        ("geometry_curves.csv", path),
        ("geometry_by_band.csv", band_geometry),
        ("length_equalization.csv", lengths),
        ("homogeneity_checks.csv", intervention),
        ("spectrum.csv", spectrum),
        ("kernel_checks.csv", checks),
        ("datasets.csv", datasets),
        ("length_summary.csv", length_summary),
        ("length_sensitivity.csv", sensitivity),
        ("plateau_explanation.csv", plateau),
        ("extreme_test_months.csv", extremes),
    ]:
        save(name, rows)
