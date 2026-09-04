import numpy as np
from scipy import stats


def pred_vs_baseline_corr_diff_test(pred, base, iv, n_bootstrap=10_000, seed=42):
    """
    Bootstrap the difference between two one-sided Spearman correlations:
      diff = r(pred, iv) - r(base, iv)
      diff = pred - baseline


    1. Randomly select len(pred) samples with replacement
    2. Compute difference in correlations
    3. Repeat 1. and 2. n_bootstrap times
    4. p-value equals the proportion of statistics on "wrong" side of 0
    Intepretation: you want to correlations differences of the true distribution to be significantly above 0.

    pred: predicted FEV1 values (FEV1%PredFT)
    base: baseline FEV1 values
    iv: IV days

    If 0 is not in the returned CI, the difference is significant.
    """
    rng = np.random.default_rng(seed)
    pred, base, iv = np.array(pred), np.array(base), np.array(iv)
    n = len(pred)

    res_pred = stats.spearmanr(pred, iv, alternative="less")
    res_base = stats.spearmanr(base, iv, alternative="less")
    obs_diff = res_pred.statistic - res_base.statistic

    # BOOTSTRAP TEST
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = (
            stats.spearmanr(pred[idx], iv[idx], alternative="less").statistic
            - stats.spearmanr(base[idx], iv[idx], alternative="less").statistic
        )
    # Compute the p-value for the correlation differences
    p_val_bootstrapped_corr_diff = bootstrap_pvalue(boot_diffs)

    # PERMUTATION TEST
    # Pre-allocate array for permuted differences
    perm_diffs = np.zeros(n_bootstrap)
    for k in range(n_bootstrap):
        # Generate boolean mask: True = swap, False = keep
        swap_mask = rng.random(n) > 0.5

        # Swap pred_i and baseline_i where mask is True
        p_perm = np.where(swap_mask, base, pred)
        b_perm = np.where(swap_mask, pred, base)

        # Calculate statistic on permuted arrays
        perm_diffs[k] = (
            stats.spearmanr(p_perm, iv).statistic
            - stats.spearmanr(b_perm, iv).statistic
        )
    # One-sided p-value: proportion of null stats >= observed stat
    p_val_perm_test_corr_diff = np.mean(perm_diffs >= obs_diff)

    return {
        "r_baseline": res_base.statistic,
        "p_baseline": res_base.pvalue,
        "r_predicted": res_pred.statistic,
        "p_predicted": res_pred.pvalue,
        "diff": obs_diff,
        "n": n,
        "diffs": boot_diffs,
        "p_val_bootstrapped_corr_diff": p_val_bootstrapped_corr_diff,
        "p_val_perm_test_corr_diff": p_val_perm_test_corr_diff,
    }


def bootstrap_pvalue(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    # proportion of bootstrap stats on the "wrong" side of 0
    p_one_sided = np.mean(diffs >= 0)  # for H1: mean < 0
    return p_one_sided
