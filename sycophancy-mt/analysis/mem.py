"""Mixed-effects regression (Day-3 stretch goal).

Headline model (the interaction term `ambiguity_z:challenge_type` IS the central test):

  shift ~ ambiguity_z * challenge_type + prompt_family + judge + (1 | seg_id) + (1 | lang_pair)

Implementation notes:
  - statsmodels MixedLM supports one grouping variable; we use seg_id as the
    random intercept and treat lang_pair as a fixed effect to keep convergence
    well-behaved on small samples.
  - If MixedLM fails to converge: drop random slope variations, then fall back
    to OLS with cluster-robust SEs by seg_id.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from .stats import CHALLENGE_KINDS


def fit_mem(long_metrics: pd.DataFrame):
    df = long_metrics[long_metrics["condition"].isin(CHALLENGE_KINDS)].copy()
    df = df.dropna(subset=["shift", "ambiguity"])
    df["ambiguity_z"] = (df["ambiguity"] - df["ambiguity"].mean()) / df["ambiguity"].std(ddof=0)

    try:
        model = smf.mixedlm(
            "shift ~ ambiguity_z * C(condition) + C(prompt_family) + C(judge) + C(lang_pair)",
            df,
            groups=df["seg_id"],
        )
        result = model.fit(method="lbfgs", reml=False)
        return result
    except Exception as e:
        print(f"MixedLM failed: {e}\nFalling back to OLS with seg_id cluster-robust SEs.")
        ols = smf.ols(
            "shift ~ ambiguity_z * C(condition) + C(prompt_family) + C(judge) + C(lang_pair)",
            df,
        ).fit(cov_type="cluster", cov_kwds={"groups": df["seg_id"]})
        return ols


def coef_table(result) -> pd.DataFrame:
    params = result.params
    se = result.bse
    pvals = result.pvalues
    return pd.DataFrame({"coef": params, "se": se, "p": pvals})
