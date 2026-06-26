import numpy as np
from scipy import stats


def bootstrap_corr_diff(x, y, z, n_bootstrap=10_000, seed=42):
    """
    Bootstrap the difference between two one-sided Spearman correlations:
      diff = r(x, z) - r(y, z)
      diff = pred - baseline

    1. Randomly select len(x) samples with replacement 
    2. Compute difference in correlations
    3. Repeat 1-2 n_bootstrap times, check if 0 is within the 95% CI of the differences
    Intepretation: you want to correlations differences of the true distribution to be significantly above 0.

    x: baseline FEV1 values
    y: predicted FEV1 values (FEV1%PredFT)
    z: IV days

    If 0 is not in the returned CI, the difference is significant.
    """
    rng = np.random.default_rng(seed)
    x, y, z = np.array(x), np.array(y), np.array(z)
    n = len(x)

    res_x = stats.spearmanr(x, z, alternative="less")
    res_y = stats.spearmanr(y, z, alternative="less")
    obs_diff = res_x.statistic - res_y.statistic

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = (
            stats.spearmanr(x[idx], z[idx], alternative="less").statistic
            - stats.spearmanr(y[idx], z[idx], alternative="less").statistic
        )

    ci_lo_95 = np.percentile(boot_diffs, 2.5)
    ci_hi_95 = np.percentile(boot_diffs, 97.5)
    ci_lo_90 = np.percentile(boot_diffs, 5.0)
    ci_hi_90 = np.percentile(boot_diffs, 95.0)

    return {
        "r_baseline": res_x.statistic,
        "p_baseline": res_x.pvalue,
        "r_predicted": res_y.statistic,
        "p_predicted": res_y.pvalue,
        "diff": obs_diff,
        "ci_lo_95": ci_lo_95,
        "ci_hi_95": ci_hi_95,
        "significant_95": not (ci_lo_95 <= 0 <= ci_hi_95),
        "ci_lo_90": ci_lo_90,
        "ci_hi_90": ci_hi_90,
        "significant_90": not (ci_lo_90 <= 0 <= ci_hi_90),
        "n": n,
    }
