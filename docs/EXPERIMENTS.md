# Experiment map

The fixed protocol is in `references/protocol.json`. It contains the complete
feature and penalty grids, seeds and kernel reference points. The measured
tables in `measurements/` are independent of generated outputs in `results/`.

| Paper component | Command | Implementation |
| --- | --- | --- |
| Data and preprocessing | `prepare` | `data.sources`, `data.inputs`, `data.target` |
| Kernel curves, plateau and row-length mechanism | `run kernel` | `kernel.main`, `models.Model` |
| Oracle gain, amplitude, alignment and elasticity | `run kernel` | `kernel.geometry` |
| Near-optimal band and static elasticity approximation | `run kernel` | `kernel.main` |
| Four training-only selectors | `run selection` | `validation.run` |
| Financial finite-feature surfaces and bands | `run surfaces` | `features.fit_surface`, `reporting.surfaces` |
| Reliability coefficient and seed variance | `run reliability` | `reliability.coefficient`, `reliability.fit_group` |
| Independent feature-budget confirmation | `run budgets` | `budgets.run` |
| High-SNR spherical linear teacher | `run synthetic` | `synthetic.run` |
| Regularization and idealized power tables | `tables` | `reporting.tables` |
| All paper figures | `figures` | `figures.draw_paper` |

## Figure inputs

Paths below are relative to `measurements/`. Figure filenames and numbers are
also recorded in `references/figures.json`.

| Figure | Output stem | CSV inputs | Plot function |
| --- | --- | --- | --- |
| 1.1 | `fig_synthetic` | `data/synthetic_surface.csv` | `synthetic_surface` |
| 1.2 | `fig_surfaces` | `data/financial_surfaces.csv`, `tables/kernel_summary.csv` | `financial_surfaces` |
| 4.1 | `fig_surfaces_more` | Same | `financial_surfaces` |
| 4.2 | `fig_ridgeless` | `tables/ridgeless_by_P.csv`, `tables/kernel_summary.csv` | `ridgeless_plateau` |
| 4.3 | `fig_profiles_kernel` | `tables/kernel_profiles.csv`, `tables/kernel_summary.csv` | `kernel_profiles` |
| 4.4 | `fig_band_limit` | `data/finite_bands.csv`, `tables/kernel_bands.csv` | `near_optimal_band` |
| 4.5 | `fig_selection` | `tables/selection.csv`, `tables/kernel_bands.csv` | `training_selection` |
| 4.6 | `fig_alignment` | `tables/geometry_curves.csv`, `tables/geometry_by_band.csv`, `tables/kernel_summary.csv` | `alignment` |
| 4.7 | `fig_parabola` | `tables/geometry_curves.csv` | `normalized_parabola` |
| 4.8 | `fig_power_law` | `tables/geometry_curves.csv`, `tables/kernel_summary.csv` | `power_law` |
| 4.9 | `fig_band_check` | `tables/kernel_bands.csv`, `tables/kernel_summary.csv` | `band_check` |
| 4.10 | `fig_reliability` | `data/reliability_transform.csv` | `reliability_constant` |

## Table inputs

| Table | Measured CSV |
| --- | --- |
| 3.1, prediction tasks | `tables/datasets.csv`, `tables/features.csv` |
| 4.1, ridgeless plateau | `tables/plateau_comparison.csv` |
| 4.2, measured excess error and label sensitivity | `tables/plateau_explanation.csv` |
| 4.3, equalized input lengths | `tables/length_equalization.csv` |
| 4.4, tuned and ridgeless kernel models | `tables/regularization_summary.csv` |
| 4.5, training-only penalty selection | `tables/selection.csv` |
| 4.6, normalized size window | Algebraic values `1 +/- sqrt(1-q)` |
| 4.7, measured and predicted penalty bands | `tables/kernel_bands.csv` |
| 4.8, feature budgets | `tables/feature_budget.csv`, `tables/feature_budget_wide.csv` |

## Numerical conventions

The forecast is `mean(y) + C_P (G_P + lambda*m*I)^(-1) J*y`, with `m=1/2`.
Only the training labels are centered. The kernel matrices are not double
centered. The same penalty therefore has the same kernel units at every P.

Feature draws are independent, and feature columns are nested across P within
each draw. The surface uses the direct mean of model losses. Near-optimal sets
use a zero-mean first-order fluctuation control to reduce Monte Carlo error.
Ridgeless points and models with fewer than 18 features use direct losses.

The near-optimal set satisfies `R2_P(lambda) >= q*R2_infinity_best`. Finite-P
boundaries are interpolated in log penalty with a shape-preserving cubic.
The kernel-limit boundaries are solved directly. The main band approximation
uses the single elasticity evaluated at the oracle center.

The four selectors are kernel moments, restricted maximum likelihood, a single
96-month validation block and past-only expanding validation. Monthly validation
starts after 120 fit observations and advances in 12-month blocks. Annual labels
have a 12-origin purge. A post-validation embargo removes no rows from past-only
fits. All transformations are refitted inside each fold.

Budget penalties are located on one half of the feature draws and evaluated on
the other, with the halves reversed and averaged. This split concerns feature
randomness, not another train/test split of the financial history. The reported
budget MCSE uses 2,000 resamples of feature draws with penalty reselection.

The frozen kappa integral uses 262,144 Gaussian probes. Monthly tasks use seed
0 and batches of 8,192, with weights generated by columns. Macro-Treasury uses
seed 17,173,100 and batches of 4,096, with weights generated by rows. Both
evaluate the same single-feature variance. These sampling conventions reproduce
the published integration values and do not change the data preprocessing.

The synthetic example uses a uniform unit sphere, a linear teacher
`sqrt(d)*x[0]`, Gaussian noise of variance `1/50`, n=300, d=30 and 4,096 test
inputs. Random-feature weights are uniform on the sphere of radius `sqrt(d)`.
It uses the generation family of Mei and Montanari (2022). Financial weights
are Gaussian. Their ReLU kernel limits coincide, but their finite-feature
sampling distributions differ.

## Provenance and checks

`references/datasets.json` identifies the raw files. The checksums of every
published CSV are in `references/measurements.json`, and those of the processed
arrays are in `references/processed_arrays.json`. `references/paper.json`
identifies the manuscript associated with the figure map.

`python -m orb_omfp verify` checks these hashes, training-only preprocessing,
task alignment, kernel reference values, penalty-band endpoints, primal/dual
ridge agreement, fixed-seed forecasts and controls, homogeneity and the exact
reliability identity. Source and document checks reject notebook files and
machine-dependent paths.
