"""Financial panels with training-fitted standardization and AR residuals."""

import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .config import DATA, REFERENCE, TASKS
from .io import processed_path, sha, sha_array

RAW = DATA / "raw"
FEATURES = [
    "excess_returns",
    "dp",
    "dy",
    "ep",
    "de",
    "bm",
    "ntis",
    "tbl",
    "lty",
    "ltr",
    "tms",
    "dfy",
    "dfr",
    "infl",
]
COLUMNS = {
    "dp": "d/p",
    "dy": "d/y",
    "ep": "e/p",
    "de": "d/e",
    "bm": "b/m",
    "ntis": "ntis",
    "tbl": "tbl",
    "lty": "lty",
    "ltr": "ltr",
    "tms": "tms",
    "dfy": "dfy",
    "dfr": "dfr",
    "infl": "infl",
}


def load_goyal_welch(path):
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["yyyymm"].astype(str), format="%Y%m"), name="month"
    )
    frame = frame.loc["1930-01-01":].copy()
    if frame.index.has_duplicates or not np.all(np.diff(frame.index.to_period("M").asi8) == 1):
        raise ValueError("Expected a complete monthly calendar")
    variables = pd.DataFrame(index=frame.index)
    variables["excess_returns"] = frame.ret - frame.Rfree
    for name, column in COLUMNS.items():
        values = pd.to_numeric(frame[column], errors="raise")
        variables[name] = np.log(values) if name in {"dp", "dy", "ep", "de"} else values
    if not np.isfinite(variables.to_numpy()).all():
        raise ValueError("Nonfinite financial predictors")
    return frame, variables


def transformed(frame, codes):
    out = {}
    for col in frame:
        x = frame[col].astype(float)
        c = int(codes[col])
        lx = np.log(x.where(x > 0))
        out[col] = {
            1: lambda: x,
            2: lambda: x.diff(),
            3: lambda: x.diff().diff(),
            4: lambda: lx,
            5: lambda: lx.diff(),
            6: lambda: lx.diff().diff(),
            7: lambda: x.pct_change(fill_method=None).diff(),
        }[c]()
    return pd.DataFrame(out, index=frame.index)


def sources():
    raw, v = load_goyal_welch(RAW / "goyal_welch.csv")
    equity = raw.ret - raw.Rfree
    bond = raw.ltr - raw.Rfree
    f = pd.read_csv(RAW / "fred_md_2026_09.csv")
    codes = f.iloc[0, 1:].astype(int)
    f = f.iloc[1:].copy()
    f.index = pd.to_datetime(f.pop("sasdate"))
    f = f.apply(pd.to_numeric).loc[:"2024-12-01"]
    fp = transformed(f, codes).shift(2)
    overlap = pd.date_range(
        max(raw.index.min(), f.index.min()), min(raw.index.max(), f.index.max()), freq="MS"
    )
    usable = overlap[5:]
    cut = 2 * len(usable) // 3
    train, test = usable[:cut], usable[cut:]
    mat = loadmat(RAW / "ludvigson_ng.mat")
    headers = [
        str(s[0]).strip() if len(s) else f"col{i}" for i, s in enumerate(mat["headertext"][0, :])
    ]
    m = pd.DataFrame(
        mat["macrodat"], index=pd.date_range("1959-01-01", periods=588, freq="MS"), columns=headers
    )
    codes = pd.Series(mat["vartype"].ravel(), index=headers).replace(0, 1)
    mp = transformed(m, codes).shift(1)
    price = pd.read_csv(
        RAW / "bondprice.dat",
        comment="%",
        sep="\\s+",
        header=None,
        names=["date", "p1", "p2", "p3", "p4", "p5"],
    )
    price.index = (
        pd.DatetimeIndex(pd.to_datetime(price.pop("date").astype(str), format="%Y%m%d"))
        .to_period("M")
        .to_timestamp()
    )
    lp = np.log(price / 100)
    target = pd.Series(
        np.column_stack(
            [(lp[f"p{j - 1}"].shift(-12) - lp[f"p{j}"] + lp.p1).to_numpy() for j in range(2, 6)]
        ).mean(1),
        index=price.index,
    ).dropna()
    tr = target.loc["1964-01-01":].index[:288]
    te = target.loc["1964-01-01":].index[300:]

    def entry(published, y, train, test, h):
        return dict(published=published, y=y, train=train, test=test, horizon=h)

    gx = v[FEATURES].shift(1)
    return {
        "gw": entry(gx, equity, train, test, 1),
        "gw_bond": entry(gx, bond, train, test, 1),
        "fred_equity": entry(fp, equity, train, test, 1),
        "macro_treasury": entry(mp, target, tr, te, 12),
    }


def ar_fit(frame, train, lag=1):
    previous = frame.shift(lag)
    a, b = (previous.loc[train], frame.loc[train])
    am, bm = (a.mean(), b.mean())
    rho = ((a - am) * (b - bm)).sum() / ((a - am) ** 2).sum()
    alpha = bm - rho * am
    return (frame - alpha - rho * previous, alpha, rho)


def inputs(source):
    train, test = source["train"], source["test"]
    # Fit column scales on the training block.
    panel = source["published"].copy().ffill()
    eligible = panel.loc[train].notna().all() & (panel.loc[train].std() > 1e-10)
    panel = panel.loc[:, eligible]
    mu, sd = panel.loc[train].mean(), panel.loc[train].std(ddof=0)
    z = (panel - mu) / sd
    # Remove first-order persistence with fixed coefficients.
    residual, alpha, rho = ar_fit(z, train)
    center = residual.loc[train].mean()
    residual = residual - center
    X, Xo = residual.loc[train].to_numpy(), residual.loc[test].to_numpy()
    scale = np.sqrt(np.mean(np.sum(X * X, axis=1)))
    X, Xo = X / scale, Xo / scale
    if not np.isfinite(np.r_[X, Xo]).all():
        raise ValueError("Nonfinite transformed inputs")
    meta = dict(
        columns=list(panel.columns),
        mean=mu.to_dict(),
        sd=sd.to_dict(),
        ar_intercept=alpha.to_dict(),
        ar_slope=rho.to_dict(),
        residual_center=center.to_dict(),
        global_RMS=float(scale),
    )
    return X, Xo, meta


def target(source):
    # Residualize in original units before the equivalent affine rescaling.
    raw, train, test = source["y"], source["train"], source["test"]
    residual, alpha, rho = ar_fit(raw, train, lag=source["horizon"])
    mu, sd = residual.loc[train].mean(), residual.loc[train].std(ddof=0)
    y, yo = (residual.loc[train].to_numpy() - mu) / sd, (residual.loc[test].to_numpy() - mu) / sd
    if not np.isfinite(np.r_[y, yo]).all():
        raise ValueError("Nonfinite transformed target")
    meta = dict(
        ar_intercept=float(alpha),
        ar_slope=float(rho),
        lag=int(source["horizon"]),
        residual_mean=float(mu),
        residual_sd=float(sd),
    )
    return y, yo, meta


def prepare():
    manifest = json.loads((REFERENCE / "datasets.json").read_text())
    for entry in manifest:
        if sha(RAW / entry["file"]) != entry["sha256"]:
            raise ValueError("Raw data checksum mismatch: " + entry["file"])
    processed_path(TASKS[0]).parent.mkdir(parents=True, exist_ok=True)
    for task, source in sources().items():
        X, Xo, xm = inputs(source)
        y, yo, ym = target(source)
        arrays = dict(
            X=X,
            Xo=Xo,
            y=y,
            yo=yo,
            train_ids=source["train"].astype(str).to_numpy(dtype=str),
            test_ids=source["test"].astype(str).to_numpy(dtype=str),
            feature_columns=np.array(xm["columns"], dtype=str),
        )
        np.savez_compressed(processed_path(task), **arrays)
        meta = dict(
            task=task,
            inputs=xm,
            target=ym,
            array_sha256={k: sha_array(v) for k, v in arrays.items()},
        )
        processed_path(task).with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(task, X.shape, Xo.shape, flush=True)
