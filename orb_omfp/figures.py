"""Paper figures from measured CSV files."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from .config import MEASUREMENTS, ROOT

ROOT_DATA = MEASUREMENTS
FIG = ROOT / "figures"

TASKS = ["gw", "gw_bond", "fred_equity", "macro_treasury"]
NAMES = {"gw": "GW1", "gw_bond": "GW2", "fred_equity": "FRED", "macro_treasury": "Macro-Treasury"}
LONG_NAMES = {
    "gw": "GW1 equity",
    "gw_bond": "GW2 bond",
    "fred_equity": "FRED equity",
    "macro_treasury": "Macro-Treasury",
}

# ---------------------------------------------------------------- style
WIDTH = 5.9  # inches, text width of the thesis
COLOR = {
    "gw": "#1d3d6e",
    "gw_bond": "#8f3424",
    "fred_equity": "#7699c4",
    "macro_treasury": "#c98a62",
}
DASHED = (0, (4.5, 1.6))
LINESTYLE = {
    "gw": "-",
    "gw_bond": "-",
    "fred_equity": DASHED,
    "macro_treasury": DASHED,
}  # dashed: macro inputs
BLACK, GREY, MID = "#1a1a1a", "#8a8a8a", "#5a5a5a"
SHADE = "#ececec"  # windows that do not belong to one task
DOTTED = (0, (1, 1.2))
TINT_Q = {
    0.90: 0.20,
    0.95: 0.40,
    0.99: 0.65,
}  # share of the task colour in band shades, the rest is white
NEUTRAL = "#4d4d4d"  # legend swatches for bands that take the colour of each task
BAND = COLOR[
    "gw"
]  # near-optimal bands in 07 and 12 share one colour: they show the same pattern on every task
METHODS = [
    ("kernel_moments", "M1 kernel moments", "o", True),
    ("reml", "M2 REML", "s", False),
    ("validation", "M3 single split", "^", True),
    ("rolling_cv", "M4 rolling CV", "D", False),
]

PAPER_RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.grid": True,
    "grid.color": "#e9e9e9",
    "grid.linewidth": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlepad": 3,
}
SURFACE_RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.transparent": False,
}
SURFACE_GRID = {"grid.color": "#b0b0b0", "grid.linewidth": 0.4, "axes.linewidth": 0.8}
SURFACE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "teal_grey_magenta",
    [
        (0, "#8d3964"),
        (0.25, "#cb8dac"),
        (0.5, "#e4e7e2"),
        (0.75, "#b8d8cd"),
        (0.95, "#6aafa0"),
        (0.975, "#2a7a6e"),
        (0.995, "#0b4c45"),
        (1, "#082f36"),
    ],
)


# ---------------------------------------------------------------- helpers
def read(name, table=False):
    return pd.read_csv(ROOT_DATA / ("tables" if table else "data") / name)


def tint(color, w):
    """Mix a colour with white: w = 1 gives the colour and w = 0 gives white."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(1 - w * (1 - c))


def plain(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def powers(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))
    axis.set_minor_locator(NullLocator())


def save(fig, out, stem, dpi=300, **kw):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=dpi, **kw)
    plt.close(fig)
    print("Plotted", stem, flush=True)


# ---------------------------------------------------------------- 01, 02, 02b error surfaces
def _surface_colorbar(fig, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(-1, 1), cmap=SURFACE_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    cb.ax.set_xticklabels([r"$\leq -1$", "0", "1"])
    cb.set_label(r"share of the best gain at $P=\infty$", fontsize=9, labelpad=1)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)


def synthetic_surface(out=FIG, stem="01_synthetic_surface"):
    """Spherical linear teacher with SNR 50."""
    sy = read("synthetic_surface.csv")
    best_nmse = float(sy[np.isinf(sy.P)].nmse.min()) if np.isinf(sy.P).any() else 0.027081
    sy = sy[np.isfinite(sy.P) & np.isfinite(sy.penalty)]
    pivot = sy.pivot(index="penalty", columns="P_over_n", values="nmse").sort_index()
    pen = pivot.index.to_numpy()
    x, y = np.meshgrid(
        np.log10(pivot.columns.to_numpy()),
        np.where(pen > 0, np.log10(np.maximum(pen, 1e-300)), -8.0),
    )
    z = np.log10(pivot.to_numpy())
    with plt.rc_context(SURFACE_RC):
        fig = plt.figure(figsize=(4.05, 3.9))
        with plt.rc_context(SURFACE_GRID):
            ax = fig.add_axes([0.065, 0.19, 0.88, 0.76], projection="3d")
        face = SURFACE_CMAP(np.clip(((1 - 10**z) / (1 - best_nmse) + 1) / 2, 0, 1))
        surface = ax.plot_surface(
            x, y, z, facecolors=face, shade=False, rcount=z.shape[0], ccount=z.shape[1], linewidth=0
        )
        surface.set_rasterized(True)
        ax.view_init(27, -130)
        ax.set_xlabel(r"complexity $\log_{10}c$", labelpad=-2)
        ax.set_ylabel(r"ridge penalty $\log_{10}\lambda$", labelpad=-1)
        ax.set_zlabel(r"$\log_{10}$ NMSE", labelpad=-2)
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_yticks([-8, -4, 0, 4])
        ax.set_yticklabels([r"$-\infty$", "$-4$", "0", "4"])
        lo, hi = min(0.0, float(z.min())), float(z.max())
        ax.set_zlim(lo, hi)
        ax.set_zticks(np.arange(np.ceil(lo), np.floor(hi) + 0.5, 1))
        ax.tick_params(axis="both", pad=-1.5, labelsize=8)
        _surface_colorbar(fig, [0.31, 0.085, 0.38, 0.022])
        # Crop the unused title strip.
        save(
            fig,
            out,
            stem,
            dpi=400,
            bbox_inches=Bbox.from_bounds(0, 0, 4.05, 3.9 * 1459 / 1560),
            facecolor="white",
        )


def financial_surfaces(tasks=("gw", "macro_treasury"), out=FIG, stem="02_financial_surfaces"):
    """Two financial tasks side by side: height log10 NMSE, colour share of the best gain at P = infinity."""
    data = read("financial_surfaces.csv")
    best = read("kernel_summary.csv", True).set_index("task").R2_star
    with plt.rc_context(SURFACE_RC):
        fig = plt.figure(figsize=(6.4, 3.7))
        for k, task in enumerate(tasks):
            d = data[(data.task == task) & (data.P >= 18) & np.isfinite(data.lambda_value)]
            d = d.assign(
                y=np.where(d.lambda_value > 0, np.log10(d.lambda_value.clip(lower=1e-300)), -8.0)
            )
            piv = d.pivot_table(index="y", columns="c", values="NMSE", aggfunc="mean").sort_index()
            piv = piv[piv.index <= 6]
            X, Y = np.meshgrid(
                np.log10(piv.columns.values.astype(float)), piv.index.values.astype(float)
            )
            z = np.log10(piv.values)
            face = SURFACE_CMAP(np.clip(((1 - 10**z) / best[task] + 1) / 2, 0, 1))
            with plt.rc_context(SURFACE_GRID):
                ax = fig.add_axes([0.075 + 0.49 * k, 0.17, 0.43, 0.79], projection="3d")
            surf = ax.plot_surface(
                X,
                Y,
                z,
                facecolors=face,
                shade=False,
                rcount=z.shape[0],
                ccount=z.shape[1],
                linewidth=0,
            )
            surf.set_rasterized(True)
            ax.view_init(27, -130)
            ax.set_title(LONG_NAMES[task], pad=0)
            ax.set_xlabel(r"complexity $\log_{10} c$", labelpad=-2)
            ax.set_ylabel(r"ridge penalty $\log_{10}\lambda$", labelpad=-1)
            ax.set_zlabel(r"$\log_{10}$ NMSE", labelpad=-3)
            ax.set_xticks([-1, 0, 1, 2, 3] if X.max() > 2.7 else [-1, 0, 1, 2])
            ax.set_yticks([-8, -4, 0, 4])
            ax.set_yticklabels([r"$-\infty$", "$-4$", "0", "4"])
            ax.set_zticks(np.arange(0, np.floor(np.nanmax(z)) + 0.5, 1))
            ax.set_zlim(min(0.0, float(np.nanmin(z))), float(np.nanmax(z)))
            ax.tick_params(axis="both", pad=-1.5, labelsize=8)
        _surface_colorbar(fig, [0.33, 0.10, 0.36, 0.022])
        save(fig, out, stem, dpi=400, bbox_inches=fig.bbox_inches, facecolor="white")


# ---------------------------------------------------------------- 03, 04 plateau and length mechanism
def ridgeless_plateau(out=FIG, stem="03_ridgeless_plateau"):
    a = read("ridgeless_by_P.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 1.62))
        for ax, t in zip(axs, TASKS):
            f = a[(a.task == t) & (a.P >= 18)].sort_values("P")
            col = COLOR[t]
            lo = np.minimum(1.96 * f.NMSE_MCSE, 0.9 * f.NMSE)
            ax.errorbar(
                f.c,
                f.NMSE,
                yerr=[lo, 1.96 * f.NMSE_MCSE],
                fmt="o-",
                ms=1.8,
                lw=0.8,
                elinewidth=0.5,
                capsize=0.8,
                color=col,
                mfc=col,
                mec=col,
            )
            ax.axhline(ref.loc[t, "ridgeless_NMSE"], color=BLACK, ls=DOTTED, lw=0.8)
            ax.axhline(1, color=GREY, lw=0.5)
            ax.axvline(1, color=GREY, ls=(0, (3, 2)), lw=0.5)
            ax.set(xscale="log", yscale="log", xlabel="$P/n$", title=NAMES[t])
            ax.xaxis.set_major_locator(LogLocator(numticks=4))
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
        axs[0].set_ylabel(r"NMSE at $\lambda=0$")
        fig.tight_layout(w_pad=0.4)
        save(fig, out, stem)


def length_equalization(out=FIG, stem="04_length_equalization"):
    f = read("length_equalization.csv", True)
    groups = [("gw", "GW inputs"), ("fred_equity", "FRED"), ("macro_treasury", "Macro-Treasury")]
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 1.9))
        for ax, (t, label) in zip(axs, groups):
            r = (
                f[f.task == t]
                .set_index("intervention")
                .loc[["original", "train_unit", "test_unit", "both_unit"]]
            )
            col = COLOR[t]
            bars = ax.bar(
                range(4),
                r.deltaL,
                color=[col, tint(col, 0.65), tint(col, 0.40), tint(col, 0.20)],
                width=0.68,
                edgecolor=col,
                linewidth=0.5,
            )
            ax.bar_label(bars, fmt="%.3f", padding=1.5, fontsize=6.5)
            ax.set_xticks(range(4), ["original", "train eq.", "test eq.", "both"])
            ax.set_ylim(0, r.deltaL.max() * 1.22)
            ax.set_title(label)
            ax.grid(axis="x", visible=False)
        axs[0].set_ylabel(r"$\Delta L(0)$")
        fig.tight_layout(w_pad=0.6)
        save(fig, out, stem)


# ---------------------------------------------------------------- 05, 06 penalty profiles
def kernel_profiles(out=FIG, stem="05_kernel_penalty_profiles"):
    a = read("kernel_profiles.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    yticks = {
        "gw": [1, 2, 4],
        "gw_bond": [1, 1.5, 2],
        "fred_equity": [1, 2, 4],
        "macro_treasury": [1, 1.2, 1.4, 1.6],
    }
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(2, 4, figsize=(WIDTH, 2.75))
        for j, t in enumerate(TASKS):
            f = a[(a.task == t) & (a.lambda_value > 0) & np.isfinite(a.lambda_value)]
            r = ref.loc[t]
            for k in range(2):
                ax = axs[k, j]
                ax.plot(f.lambda_value, f.NMSE, color=COLOR[t], lw=1.0)
                ax.axhline(1, color=GREY, lw=0.5)
                ax.axvline(r.lambda_star, color=BLACK, ls=DOTTED, lw=0.7)
                ax.plot([r.lambda_star], [1 - r.R2_star], marker="D", ms=2.6, color=BLACK, zorder=4)
                ax.set_xscale("log")
                powers(ax.xaxis, [1e-6, 1e-2, 1e2, 1e6])
            axs[0, j].set_yscale("log")
            plain(axs[0, j].yaxis, yticks[t])
            axs[0, j].yaxis.set_minor_locator(NullLocator())
            axs[0, j].set_title(NAMES[t])
            axs[1, j].set_ylim(1 - r.R2_star * 1.3, 1 + r.R2_star * 0.6)
            axs[1, j].text(
                0.04,
                0.08,
                rf"$R^2_*={100 * r.R2_star:.3f}\%$",
                ha="left",
                transform=axs[1, j].transAxes,
                fontsize=6.5,
            )
            axs[1, j].set_xlabel(r"$\lambda$")
            axs[1, j].yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip("."))
            )
        axs[0, 0].set_ylabel("NMSE")
        axs[1, 0].set_ylabel("NMSE (zoom)")
        fig.tight_layout(w_pad=0.3, h_pad=0.4)
        save(fig, out, stem)


def profiles_many_p(out=FIG, stem="06_penalty_profiles_many_P"):
    """Direct random-feature means near P/n = 0.5, 2 and 200,000/n, against the exact curve at P = infinity."""
    a = read("financial_surfaces.csv")
    kp = read("kernel_profiles.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 2.05))
        shades = (0.35, 0.65, 1.0)
        for ax, t in zip(axs, TASKS):
            part = a[a.task == t]
            r = ref.loc[t]
            available = part.P.unique()
            ps = [
                int(available[np.abs(available - v).argmin()]) for v in (r.n / 2, 2 * r.n, 200000)
            ]
            for P, w in zip(ps, shades):
                f = part[(part.P == P) & (part.lambda_value > 0) & np.isfinite(part.lambda_value)]
                ax.plot(f.lambda_value, f.NMSE, color=tint(COLOR[t], w), lw=1.0)
            f = kp[(kp.task == t) & (kp.lambda_value > 0) & np.isfinite(kp.lambda_value)]
            ax.plot(f.lambda_value, f.NMSE, color=BLACK, ls=(0, (4, 2)), lw=0.9)
            ax.axhline(1, color=GREY, lw=0.5)
            ax.set(
                xscale="log",
                xlim=(r.lambda_star / 12, r.lambda_star * 35),
                ylim=(1 - r.R2_star * 1.35, 1 + r.R2_star * 0.75),
                xlabel=r"$\lambda$",
                title=NAMES[t],
            )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").rstrip("."))
            )
        axs[0].set_ylabel("NMSE (zoom)")
        labels = [r"$P\approx n/2$", r"$P\approx 2n$", r"$P=200{,}000$"]
        handles = [
            Line2D([], [], color=tint(NEUTRAL, w), lw=1.0, label=lab)
            for w, lab in zip(shades, labels)
        ]
        handles += [Line2D([], [], color=BLACK, ls=(0, (4, 2)), lw=0.9, label=r"$P=\infty$")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.3)
        save(fig, out, stem)


# ---------------------------------------------------------------- 07, 08 near-optimal band and its center
def near_optimal_band(out=FIG, stem="07_empirical_near_optimal_band"):
    a = read("finite_bands.csv")
    b = read("kernel_bands.csv", True)
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(2, 2, figsize=(5.0, 3.75))
        for ax, t in zip(axs.flat, TASKS):
            f = a[(a.task == t) & (a.P >= 18)].sort_values("P")
            center = f.kernel_lambda_star.iloc[0]
            cmax = f.c.max()
            for q, w in TINT_Q.items():
                part = f[np.isclose(f.q, q)]
                ax.fill_between(
                    part.c,
                    part.empirical_lower / center,
                    part.empirical_upper / center,
                    color=tint(BAND, w),
                    lw=0,
                )
                r = b[(b.task == t) & np.isclose(b.q, q)].iloc[0]
                for edge in (r.empirical_lower, r.empirical_upper):  # the band at P = infinity
                    ax.plot(
                        [cmax * 1.3, cmax * 1.75],
                        [edge / center] * 2,
                        color=BLACK,
                        lw=1.0,
                        solid_capstyle="butt",
                    )
            g = f[np.isclose(f.q, 0.95)]
            ax.plot(g.c, g.lambda_measured / center, "o-", color=BLACK, ms=1.8, lw=0.8)
            ax.set(xscale="log", yscale="log", ylim=(0.4, 3.3), title=NAMES[t])
            plain(ax.yaxis, [0.5, 1, 2, 3])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(LogLocator(numticks=5))
            ax.xaxis.set_minor_locator(NullLocator())
        for ax in axs[1]:
            ax.set_xlabel("$P/n$")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$\lambda/\lambda^*$")
        handles = [Patch(color=tint(BAND, w), label=f"$q={q:.2f}$") for q, w in TINT_Q.items()]
        handles += [
            Line2D([], [], color=BLACK, marker="o", ms=2.5, lw=0.8, label=r"measured $\lambda^*_P$")
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.8)
        save(fig, out, stem)


def training_selection(out=FIG, stem="08_training_only_selection"):
    """Penalties selected from the training history (M1-M4) relative to the oracle center."""
    sel = read("selection.csv", True)
    bands = read("kernel_bands.csv", True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(4.9, 2.55))
        rows = {t: 3 - i for i, t in enumerate(TASKS)}
        for t in TASKS:
            y, col = rows[t], COLOR[t]
            for q in (0.90, 0.95):
                r = bands[(bands.task == t) & np.isclose(bands.q, q)].iloc[0]
                lo, hi = r.empirical_lower / r.lambda_star, r.empirical_upper / r.lambda_star
                ax.add_patch(
                    plt.Rectangle(
                        (lo, y - 0.36), hi - lo, 0.72, color=tint(col, TINT_Q[q]), lw=0, zorder=1
                    )
                )
            ax.plot([1, 1], [y - 0.36, y + 0.36], color=BLACK, lw=0.8, zorder=2)
            for k, (method, _, marker, filled) in enumerate(METHODS):
                r = sel[(sel.task == t) & (sel.method == method)].iloc[0]
                yy = y + (1.5 - k) * 0.17
                ratio = float(r.lambda_ratio)
                if np.isfinite(ratio):
                    ax.plot(
                        ratio,
                        yy,
                        marker,
                        ms=4.2,
                        color=col,
                        mfc=col if filled else "white",
                        mec="white" if filled else col,
                        mew=0.7,
                        zorder=4,
                    )
                else:  # the method selects lambda = infinity, that is, the mean forecast
                    ax.annotate(
                        "",
                        xy=(58, yy),
                        xytext=(37, yy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=7),
                        zorder=4,
                    )
                    ax.text(34, yy, r"$\infty$", color=col, ha="right", va="center", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 60)
        plain(ax.xaxis, [0.1, 0.3, 1, 3, 10, 30])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks([rows[t] for t in TASKS])
        ax.set_yticklabels([NAMES[t] for t in TASKS])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"selected penalty relative to the optimal penalty, $\hat\lambda/\lambda^*$")
        handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=BLACK if filled else "white",
                markeredgecolor=BLACK,
                ms=4.5,
                label=label,
            )
            for _, label, marker, filled in METHODS
        ]
        handles += [
            Patch(color=tint(NEUTRAL, TINT_Q[0.95]), label="95% band"),
            Patch(color=tint(NEUTRAL, TINT_Q[0.90]), label="90% band"),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            handletextpad=0.3,
            columnspacing=1.0,
        )
        save(fig, out, stem)


# ---------------------------------------------------------------- 09-12 geometry of the band
def alignment(out=FIG, stem="09_alignment_in_90_gain_band"):
    a = read("geometry_curves.csv", True)
    b = read("geometry_by_band.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(2, 2, figsize=(4.6, 3.3))
        for ax, t in zip(axs.flat, TASKS):
            r = b[(b.task == t) & np.isclose(b.q, 0.90)].iloc[0]
            f = a[(a.task == t) & a.lambda_value.between(r.lower, r.upper)].sort_values("a")
            ax.plot(f.a, 100 * f.cos, color=COLOR[t], lw=1.1)
            ax.plot(
                f.a,
                100 * (r.alignment_intercept + r.alignment_slope * f.a),
                color=BLACK,
                ls=(0, (4, 2)),
                lw=0.8,
            )
            g = a[a.task == t]
            center = g.iloc[np.abs(g.lambda_ratio - 1).argmin()]
            ax.axhline(100 * center.cos, color=GREY, ls=DOTTED, lw=0.6)
            ax.set_title(NAMES[t])
            ax.text(
                0.96,
                0.07 if r.alignment_slope > 0 else 0.86,
                rf"$\beta_{{\mathrm{{MZ}}}}={ref.loc[t, 'beta_MZ']:.2f}$",
                ha="right",
                transform=ax.transAxes,
                fontsize=7,
            )
        for ax in axs[1]:
            ax.set_xlabel("$a$")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$\cos\theta$ (%)")
        fig.tight_layout(h_pad=0.6, w_pad=0.8)
        save(fig, out, stem)


def normalized_parabola(out=FIG, stem="10_normalized_gain_parabola"):
    a = read("geometry_curves.csv", True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(4.0, 2.55))
        t = np.linspace(0, 2.05, 301)
        ax.axvspan(1 - np.sqrt(0.05), 1 + np.sqrt(0.05), color=SHADE, lw=0, zorder=0)
        ax.plot(t, 2 * t - t * t, color=BLACK, lw=1.5, label="$2t-t^2$", zorder=2)
        for task in TASKS:
            f = a[(a.task == task) & (a.t <= 2.05) & (a.retained_gain >= -1.2)].sort_values("t")
            ax.plot(
                f.t,
                f.retained_gain,
                color=COLOR[task],
                ls=LINESTYLE[task],
                lw=1.0,
                label=NAMES[task],
                zorder=3,
            )
        ax.axhline(0.95, color=GREY, ls=(0, (3, 2)), lw=0.6)
        ax.set(
            xlim=(0, 2.05),
            ylim=(-1.2, 1.15),
            xlabel=r"$t=\|p(\lambda)\|/\|p(\lambda^*)\|$",
            ylabel=r"$R^2(\lambda)/R^2_*$",
        )
        ax.legend(ncol=2, loc="lower center", handlelength=2.6)
        save(fig, out, stem)


def power_law(out=FIG, stem="11_power_law_mapping"):
    a = read("geometry_curves.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 1.62))
        for ax, t in zip(axs, TASKS):
            f = a[(a.task == t) & a.lambda_ratio.between(0.28, 3.6)]
            ax.axhspan(1 - np.sqrt(0.05), 1 + np.sqrt(0.05), color=SHADE, lw=0)
            ax.plot(f.lambda_ratio, f.t, color=COLOR[t], lw=1.1)
            ax.plot(f.lambda_ratio, f.power_t, color=BLACK, ls=(0, (4, 2)), lw=0.8)
            ax.set(
                xscale="log",
                yscale="log",
                ylim=(0.45, 1.85),
                xlabel=r"$\lambda/\lambda^*$",
                title=NAMES[t],
            )
            plain(ax.xaxis, [0.5, 1, 2])
            ax.xaxis.set_minor_locator(NullLocator())
            plain(ax.yaxis, [0.5, 0.75, 1, 1.5])
            ax.yaxis.set_minor_locator(NullLocator())
            ax.text(
                0.96,
                0.9,
                rf"$\omega^*={ref.loc[t, 'omega_star']:.2f}$",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
            )
        axs[0].set_ylabel("$t$")
        fig.tight_layout(w_pad=0.4)
        save(fig, out, stem)


def band_check(out=FIG, stem="12_measured_and_predicted_bands"):
    b = read("kernel_bands.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(4.4, 2.7))
        labels, ticks = [], []
        for j, task in enumerate(TASKS):
            center = ref.loc[task, "lambda_star"]
            for k, q in enumerate([0.99, 0.95, 0.9]):
                r = b[(b.task == task) & np.isclose(b.q, q)].iloc[0]
                y = -j * 1.35 - k * 0.32
                ax.add_patch(
                    plt.Rectangle(
                        (r.empirical_lower / center, y - 0.12),
                        (r.empirical_upper - r.empirical_lower) / center,
                        0.24,
                        color=tint(BAND, TINT_Q[0.95]),
                        lw=0,
                    )
                )
                ax.plot(
                    [r.predicted_lower / center, r.predicted_upper / center],
                    [y, y],
                    color=BLACK,
                    lw=1.0,
                    marker="|",
                    ms=6,
                    mew=1.0,
                )
                ax.text(0.345, y, f"{100 * q:.0f}%", fontsize=6.5, va="center", color=MID)
            ticks.append(-j * 1.35 - 0.32)
            labels.append(NAMES[task])
        ax.axvline(1, color=BLACK, ls=DOTTED, lw=0.7)
        ax.set(
            xscale="log",
            xlim=(0.33, 3.3),
            yticks=ticks,
            yticklabels=labels,
            xlabel=r"$\lambda/\lambda^*$",
        )
        plain(ax.xaxis, [0.5, 0.75, 1, 1.5, 2, 3])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.legend(
            handles=[
                Patch(color=tint(BAND, TINT_Q[0.95]), label="measured"),
                Line2D(
                    [], [], color=BLACK, marker="|", ms=6, lw=1.0, label=r"band from $\omega^*$"
                ),
            ],
            loc="upper right",
        )
        save(fig, out, stem)


# ---------------------------------------------------------------- 13-16 reliability, budget, elasticity
def reliability_constant(out=FIG, stem="13_reliability_constant"):
    a = read("reliability_transform.csv")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 1.58))
        for ax, t in zip(axs, TASKS):
            f = a[a.task == t]
            k = f.kappa.iloc[0]
            col = COLOR[t]
            ax.errorbar(
                f.P,
                f.transformed,
                yerr=1.96 * f.transformed_SE,
                fmt="o",
                ms=1.9,
                capsize=0.8,
                color=col,
                elinewidth=0.5,
                mec=col,
                mfc=col,
            )
            ax.axhline(k, color=BLACK, ls=(0, (4, 2)), lw=0.8)
            ax.text(
                0.96,
                0.92,
                rf"$\kappa={k:.2f}$" if k < 100 else rf"$\kappa={k:.1f}$",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
            )
            ax.set(xscale="log", ylim=(0, k * 2.1), xlabel="$P$", title=NAMES[t])
            powers(ax.xaxis, [1e2, 1e3, 1e4, 1e5])
        axs[0].set_ylabel(r"$P\,(1/\tilde\eta_P-1)$")
        fig.tight_layout(w_pad=0.4)
        save(fig, out, stem)


def reliability_vs_gain(out=FIG, stem="14_reliability_vs_gain"):
    """Retained share with the penalty retuned, against P/(P+kappa) and the calibrated (2 beta_MZ - 1) form."""
    a = read("budget_curves.csv")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 1.75))
        x = np.geomspace(1.7, 2000, 301)
        for ax, t in zip(axs, TASKS):
            f = a[a.task == t].sort_values("P")
            k, gam = f.kappa.iloc[0], f.gamma.iloc[0]
            col = COLOR[t]
            ax.plot(x, x / (1 + x), color=BLACK, ls=(0, (4, 2)), lw=0.8)
            ax.plot(x, x / (x + 1 / gam), color=col, lw=1.0)
            ax.errorbar(
                f.P / k,
                f.eta_tuned,
                yerr=1.96 * f.eta_bootstrap_SE,
                fmt="o",
                ms=1.9,
                capsize=0.8,
                color=col,
                elinewidth=0.5,
                mec=col,
                mfc=col,
            )
            ax.set(
                xscale="log",
                ylim=(0.4, 1.025),
                xlim=(1.7, 2000),
                xlabel=r"$P/\kappa$",
                title=NAMES[t],
            )
            powers(ax.xaxis, [1e1, 1e2, 1e3])
        axs[0].set_ylabel(r"retained share $\eta_P$")
        fig.tight_layout(w_pad=0.4)
        save(fig, out, stem)


def budget_check(out=FIG, stem="15_budget_check"):
    a = read("feature_budget.csv", True)
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 1.9))
        for ax, q in zip(axs, [0.95, 0.99]):
            for j, t in enumerate(TASKS):
                r = a[(a.task == t) & np.isclose(a.q, q)].iloc[0]
                y = 3 - j
                ax.errorbar(
                    r.retained_percent,
                    y,
                    xerr=1.96 * r.MCSE_percentage_points,
                    fmt="o",
                    ms=3,
                    color=COLOR[t],
                    capsize=1.5,
                    elinewidth=0.7,
                )
                ax.annotate(
                    f"$P={int(r.P):,}$".replace(",", "{,}"),
                    (r.retained_percent, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.5,
                )
            ax.axvline(100 * q, color=GREY, ls=(0, (3, 2)), lw=0.7)
            ax.set(
                title=f"target $q={q:.2f}$",
                yticks=[3, 2, 1, 0],
                yticklabels=[NAMES[t] for t in TASKS],
                xlabel=r"retained share of $R^2_*$ (%)",
                ylim=(-0.5, 3.6),
            )
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
        fig.tight_layout(w_pad=1.0)
        save(fig, out, stem)


def elasticity_path(out=FIG, stem="16_elasticity_path"):
    a = read("geometry_curves.csv", True)
    ref = read("kernel_summary.csv", True).set_index("task")
    with plt.rc_context(PAPER_RC):
        fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 1.62))
        for ax, t in zip(axs, TASKS):
            f = a[(a.task == t) & a.lambda_ratio.between(0.25, 4)]
            ax.plot(f.lambda_ratio, f.omega, color=COLOR[t], lw=1.1)
            ax.axvline(1, color=BLACK, ls=DOTTED, lw=0.7)
            ax.axhline(ref.loc[t, "omega_star"], color=GREY, ls=DOTTED, lw=0.6)
            ax.set(xscale="log", xlabel=r"$\lambda/\lambda^*$", title=NAMES[t])
            plain(ax.xaxis, [0.25, 0.5, 1, 2, 4])
            ax.xaxis.set_minor_locator(NullLocator())
        axs[0].set_ylabel(r"$\omega(\lambda)$")
        fig.tight_layout(w_pad=0.4)
        save(fig, out, stem)


def draw_paper(source=MEASUREMENTS, out=FIG, only=None):
    global ROOT_DATA
    from .io import sha, dump

    ROOT_DATA = Path(source)
    records = json.loads((ROOT / "references/figures.json").read_text())
    outputs = []
    for r in records:
        if only and r["number"] not in only:
            continue
        function = globals()[r["function"]]
        if r["name"] == "fig_surfaces":
            function(("gw", "macro_treasury"), out, r["name"])
        elif r["name"] == "fig_surfaces_more":
            function(("gw_bond", "fred_equity"), out, r["name"])
        else:
            function(out, r["name"])
        outputs.append(
            dict(
                **r,
                PNG_sha256=sha(Path(out) / (r["name"] + ".png")),
                PDF_sha256=sha(Path(out) / (r["name"] + ".pdf")),
            )
        )
    inputs = {str(p.relative_to(ROOT_DATA)): sha(p) for p in ROOT_DATA.rglob("*.csv")}
    dump(
        Path(out) / "manifest.json",
        dict(
            generator="orb_omfp.figures.draw_paper",
            code_sha256=sha(Path(__file__)),
            inputs_sha256=inputs,
            figures=outputs,
        ),
    )


# ---------------------------------------------------------------- groups (names used in tables/asset_index.csv)
