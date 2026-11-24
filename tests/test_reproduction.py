"""Independent numerical and provenance checks for paper reproduction."""

import ast
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from orb_omfp.config import ROOT, DATA, REFERENCE, MEASUREMENTS, TASKS, anchor
from orb_omfp.data import sources, inputs, target
from orb_omfp.io import data, sha, sha_array
from orb_omfp.models import Model


class ReproductionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = {t: data(t) for t in TASKS}
        cls.models = {t: Model(z["X"], z["Xo"], z["y"]) for t, z in cls.tasks.items()}

    def test_raw_and_measurement_checksums(self):
        for r in json.loads((REFERENCE / "datasets.json").read_text()):
            self.assertEqual(sha(DATA / "raw" / r["file"]), r["sha256"])
        for r in json.loads((REFERENCE / "measurements.json").read_text()):
            self.assertEqual(sha(ROOT / r["file"]), r["sha256"])

    def test_processed_checksums(self):
        refs = json.loads((REFERENCE / "processed_arrays.json").read_text())
        for task, records in refs.items():
            for k, v in records.items():
                self.assertEqual(sha_array(self.tasks[task][k]), v)

    def test_task_alignment(self):
        for task, d, n, no in [
            ("gw", 14, 524, 263),
            ("gw_bond", 14, 524, 263),
            ("fred_equity", 116, 524, 263),
            ("macro_treasury", 131, 288, 168),
        ]:
            z = self.tasks[task]
            self.assertEqual(z["X"].shape, (n, d))
            self.assertEqual(z["Xo"].shape, (no, d))
            assert_allclose(z["X"].mean(0), 0, atol=1e-13)
            assert_allclose(np.mean(np.sum(z["X"] ** 2, 1)), 1, atol=1e-13)
        for k in ["X", "Xo"]:
            assert_allclose(self.tasks["gw"][k], self.tasks["gw_bond"][k], atol=0)
        for k in ["y", "yo"]:
            assert_allclose(self.tasks["gw"][k], self.tasks["fred_equity"][k], atol=0)

    def test_holdout_does_not_change_preprocessing_fit(self):
        for s in sources().values():
            changed = dict(s, published=s["published"].copy(), y=s["y"].copy())
            changed["published"].loc[s["test"]] *= 17
            changed["y"].loc[s["test"]] += 3
            X, _, xm = inputs(s)
            X2, _, xm2 = inputs(changed)
            y, _, ym = target(s)
            y2, _, ym2 = target(changed)
            assert_allclose(X, X2, atol=0)
            assert_allclose(y, y2, atol=0)
            self.assertEqual(xm, xm2)
            self.assertEqual(ym, ym2)

    def test_kernel_and_band_measurements(self):
        reference = pd.read_csv(MEASUREMENTS / "tables/kernel_summary.csv").set_index("task")
        bands = pd.read_csv(MEASUREMENTS / "tables/kernel_bands.csv")
        from orb_omfp.kernel import geometry

        for task, z in self.tasks.items():
            m = self.models[task]
            r = reference.loc[task]
            lam = anchor(task)["lambda_star"]
            e = z["yo"] - z["y"].mean()
            D = np.mean(e * e)
            assert_allclose(np.trace(m.G) / m.n, 0.5, atol=1e-13)
            assert_allclose(1 - m.loss(lam, z["yo"])[0] / D, r.R2_star, atol=1e-12)
            assert_allclose(m.loss(0, z["yo"])[0] / D, r.ridgeless_NMSE, rtol=1e-9)
            g, _, _ = geometry(m, e, lam)
            assert_allclose(g["omega"][0], r.omega_star, rtol=1e-6)
            for b in bands[bands.task == task].itertuples():
                for edge in [b.empirical_lower, b.empirical_upper]:
                    assert_allclose(1 - m.loss(edge, z["yo"])[0] / D, b.q * r.R2_star, atol=1e-12)

    def test_independent_fixed_seed_forecasts_and_controls(self):
        refs = pd.read_csv(REFERENCE / "solver_checks.csv")
        for r in refs.itertuples():
            z = self.tasks[r.task]
            m = self.models[r.task]
            n = len(z["y"])
            W = np.random.default_rng(r.seed).normal(size=(r.P, z["X"].shape[1])).T
            F, Fo = np.maximum(z["X"] @ W, 0), np.maximum(z["Xo"] @ W, 0)
            gp, cp = F @ F.T / r.P, Fo @ F.T / r.P
            q = z["y"] - z["y"].mean()
            e = z["yo"] - z["y"].mean()
            Q = np.linalg.inv(m.G + 0.5 * r.lambda_value * np.eye(n))
            a = Q @ q
            K = m.C @ Q
            p = cp @ np.linalg.solve(gp + 0.5 * r.lambda_value * np.eye(n), q)
            h = (cp - m.C) @ a - K @ (gp - m.G) @ a
            assert_allclose((2 * e @ p - p @ p) / (e @ e), r.R2, atol=2e-11)
            assert_allclose(2 * (e - m.C @ a) @ h / (e @ e), r.control, atol=2e-11)

    def test_primal_dual_and_length_identity(self):
        z = self.tasks["gw"]
        m = self.models["gw"]
        n = m.n
        P = 18
        W = np.random.default_rng(43).normal(size=(P, z["X"].shape[1]))
        F, Fo = np.maximum(z["X"] @ W.T, 0), np.maximum(z["Xo"] @ W.T, 0)
        q = z["y"] - z["y"].mean()
        lam = 47.0
        p = Fo @ np.linalg.solve(F.T @ F + P * 0.5 * lam * np.eye(P), F.T @ q)
        d = (Fo @ F.T / P) @ np.linalg.solve(F @ F.T / P + 0.5 * lam * np.eye(n), q)
        assert_allclose(p, d, atol=1e-11)
        X = z["X"].copy()
        X[31] *= 2
        actual = Model(X, z["Xo"], z["y"]).operator(0)
        expected = m.operator(0)
        expected[:, 31] /= 2
        assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)

    def test_reliability_identity_and_budget_units(self):
        rng = np.random.default_rng(43)
        p = rng.normal(size=(32, 17))
        e = rng.normal(size=17)
        mean = p.mean(0)
        reliability = (mean @ mean) / np.mean(np.sum(p * p, 1))
        a = np.linalg.norm(mean) / np.linalg.norm(e)
        c = e @ mean / (np.linalg.norm(e) * np.linalg.norm(mean))
        assert_allclose(
            np.mean((2 * p @ e - np.sum(p * p, 1)) / (e @ e)),
            2 * a * c - a * a / reliability,
            atol=1e-14,
        )
        b = pd.read_csv(MEASUREMENTS / "tables/feature_budget.csv")
        assert_allclose(b.P, np.ceil(b.kappa * b.q / (1 - b.q)))
        assert_allclose(b.retained_percent, 100 * b.eta_tuned)
        assert_allclose(b.MCSE_percentage_points, 100 * b.eta_bootstrap_SE)
        self.assertTrue(b.maps.eq(128).all())

    def test_analytic_power_benchmark(self):
        from orb_omfp.config import Q

        for q in Q:
            lo, hi = 1 - np.sqrt(1 - q), 1 + np.sqrt(1 - q)
            assert_allclose([2 * lo - lo * lo, 2 * hi - hi * hi], [q, q], atol=1e-14)

    def test_source_and_document_portability(self):
        for folder in ["orb_omfp", "docs", "references"]:
            for p in (ROOT / folder).rglob("*"):
                if not p.is_file() or p.suffix not in [".py", ".md", ".json", ".csv"]:
                    continue
                text = p.read_text()
                self.assertFalse(any("\u3400" <= c <= "\u9fff" for c in text), p)
                if p.suffix == ".py":
                    tree = ast.parse(text)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Constant) and isinstance(node.value, str):
                            self.assertFalse(node.value.startswith(str(Path.home()) + "/"), p)
                    self.assertNotIn("sys.path", text)
        self.assertFalse(list(ROOT.rglob("*.ipynb")))


if __name__ == "__main__":
    unittest.main()
