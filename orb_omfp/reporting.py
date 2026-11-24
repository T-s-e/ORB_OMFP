"""Export measured surfaces, near-optimal sets and paper tables."""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from .config import RESULTS, Q
from .io import save_table


def intervals(penalties, gain, threshold):
    keep = np.isfinite(penalties) & (penalties > 0)
    x = np.log(penalties[keep])
    y = gain[keep] - threshold
    fit = PchipInterpolator(x, y)
    roots = [brentq(fit, x[i], x[i + 1]) for i in np.flatnonzero(y[:-1] * y[1:] < 0)]
    bounds = [x[0], *roots, x[-1]]
    return [
        (np.exp(a), np.exp(b)) for a, b in zip(bounds[:-1], bounds[1:]) if fit((a + b) / 2) >= 0
    ]


def surfaces():
    refs = pd.read_csv(RESULTS / "tables/kernel_summary.csv").set_index("task")
    frames = [pd.read_csv(p) for p in sorted((RESULTS / "surface_tables").glob("*.csv"))]
    save_table("financial_surfaces.csv", pd.concat(frames), folder="data")
    rows = []
    for path in sorted((RESULTS / "arrays").glob("surface_*.npz")):
        with np.load(path) as a:
            for ti, task in enumerate(a["tasks"]):
                ref = refs.loc[task]
                for pi, P in enumerate(a["P"]):
                    risk = a["controlled"][:, pi, :, ti].mean(0)
                    gain = 1 - risk / a["D"][ti]
                    j = int(risk.argmin())
                    for q in Q:
                        regions = intervals(a["lambda_value"], gain, q * ref.R2_star)
                        rows.append(
                            dict(
                                task=task,
                                P=int(P),
                                c=P / ref.n,
                                q=q,
                                empirical_exists=bool(regions),
                                components=len(regions),
                                empirical_lower=regions[0][0] if regions else np.nan,
                                empirical_upper=regions[-1][1] if regions else np.nan,
                                empirical_eta=gain[j] / ref.R2_star,
                                lambda_measured=a["lambda_value"][j],
                                kernel_lambda_star=ref.lambda_star,
                                n=ref.n,
                                maps=len(a["loss"]),
                                integration_estimator="zero-mean first-order loss control"
                                if P >= 18
                                else "direct mean",
                            )
                        )
    save_table("finite_bands.csv", rows, folder="data")
    frame = pd.concat(frames)
    save_table("ridgeless_by_P.csv", frame[frame.lambda_value == 0])


def tables():
    k = pd.read_csv(RESULTS / "tables/kernel_summary.csv").set_index("task")
    k["R2_ridgeless_percent"] = 100 * (1 - k.ridgeless_NMSE)
    k["R2_tuned_percent"] = 100 * k.R2_star
    save_table("regularization_summary.csv", k.reset_index())
    power = k[["R2_star", "no"]].copy()
    power["n80_independent_gaussian"] = 4 * (1.96 + 0.84) ** 2 * (1 - power.R2_star) / power.R2_star
    save_table("power.csv", power.reset_index())
    path = RESULTS / "tables/selection.csv"
    if path.exists():
        s = pd.read_csv(path).pivot(index="task", columns="method", values="R2_percent")
        s["oracle"] = 100 * k.R2_star
        save_table("selection_wide.csv", s.reset_index())
