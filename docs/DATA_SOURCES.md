# Data sources

The four frozen source files are included in `data/raw/`. Their sizes, source
locations and SHA-256 checksums are recorded in `references/datasets.json`.
The experiment ends in December 2024 for the three monthly tasks.

- **Goyal and Welch monthly predictors, 2024 update.** `goyal_welch.csv`
  provides stock returns, risk-free returns, long-bond returns and predictors.
  GW1 predicts `ret - Rfree` and GW2 predicts `ltr - Rfree`. The shared inputs
  are the lagged equity excess return and 13 classic predictors. Valuation
  ratios use their logarithmic definitions. Source: Welch and Goyal (2008),
  *A Comprehensive Look at the Empirical Performance of Equity Premium
  Prediction*, Review of Financial Studies.
- **FRED-MD, September 2026 vintage.** `fred_md_2026_09.csv` includes the
  prescribed transformation codes. Eligibility is determined using training
  observations only, leaving 116 inputs. FRED predicts the same monthly equity
  target as GW1. Source: McCracken and Ng (2016), *FRED-MD: A Monthly Database
  for Macroeconomic Research*, Journal of Business & Economic Statistics.
- **Ludvigson and Ng macroeconomic panel.** `ludvigson_ng.mat` contains 131
  series and their transformation codes. The replication archive and its
  member name are identified in the manifest. Source: Ludvigson and Ng
  (2009), *Macro Factors in Bond Risk Premia*, Review of Financial Studies.
- **Cochrane and Piazzesi bond prices.** `bondprice.dat` provides the prices
  used to construct the annual Treasury target. Source: Cochrane and Piazzesi
  (2005), *Bond Risk Premia*, American Economic Review.

For each maturity j=2,...,5, the excess log holding return at origin t is
`log(P[j-1,t+12]) - log(P[j,t]) + log(P[1,t])`. The Macro-Treasury target
averages these four returns. Training origins end in 1987, the 1988 origins
are omitted, and testing begins in 1989, after all training labels have been
realized. Test origins end in December 2002.

GW and FRED use the common calendar, reserve five initial months for
transformations and lags, and split the 787 usable months chronologically in
the ratio 524:263. GW inputs are shifted one target month and FRED inputs two
target months to allow for release timing. The Treasury macro inputs are
shifted one forecast-origin month.

Missing inputs are carried forward from past observations. Column eligibility,
means, standard deviations, autoregressive coefficients and the common input
scale are fitted on training observations. Monthly inputs use a one-month AR
lag. The Treasury target uses a twelve-month lag so the lagged annual outcome
is observable at the forecast origin. Every target remains continuous.

FRED and Treasury use revised histories. Their measurements are pseudo out of
sample, not real-time vintage backtests. Public availability does not transfer
ownership or remove the source providers' usage terms.
