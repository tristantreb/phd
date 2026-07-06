import numpy as np
from scipy import stats


def bootstrap_corr_diff(pred, base, iv, n_bootstrap=10_000, seed=42):
    """
    Bootstrap the difference between two one-sided Spearman correlations:
      diff = r(pred, iv) - r(base, iv)
      diff = pred - baseline

    1. Randomly select len(pred) samples with replacement
    2. Compute difference in correlations
    3. Repeat 1-2 n_bootstrap times, check if 0 is within the 95% CI of the differences
    Intepretation: you want to correlations differences of the true distribution to be significantly above 0.

    pred: predicted FEV1 values (FEV1%PredFT)
    base: baseline FEV1 values
    iv: IV days

    If 0 is not in the returned CI, the difference is significant.
    """
    rng = np.random.default_rng(seed)
    pred, base, iv = np.array(pred), np.array(base), np.array(iv)
    n = len(pred)

    res_x = stats.spearmanr(pred, iv, alternative="less")
    res_y = stats.spearmanr(base, iv, alternative="less")
    obs_diff = res_x.statistic - res_y.statistic

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = (
            stats.spearmanr(pred[idx], iv[idx], alternative="less").statistic
            - stats.spearmanr(base[idx], iv[idx], alternative="less").statistic
        )

 
    # Compute the p-value for the correlation differences
    p_val_corr_diffs = bootstrap_pvalue(boot_diffs)

    return {
        "r_baseline": res_x.statistic,
        "p_baseline": res_x.pvalue,
        "r_predicted": res_y.statistic,
        "p_predicted": res_y.pvalue,
        "diff": obs_diff,
        "n": n,
        "diffs": boot_diffs,
        "p_val_corr_diffs": p_val_corr_diffs
    }


def bootstrap_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    # proportion of bootstrap stats on the "wrong" side of 0
    p_one_sided = np.mean(diffs >= 0)   # for H1: mean < 0
    return p_one_sided