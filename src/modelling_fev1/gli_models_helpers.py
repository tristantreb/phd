import numpy as np


def get_lms_pred_value_for_zscore(zscore_arr, M, S, L, debug=False):
    x = np.exp(np.log(1 + zscore_arr * L * S) / L + np.log(M))
    if debug:
        print(
            f"LMS pred for zscore {zscore_arr} = exp(ln(1 + {zscore_arr} * {L} * {S}) / {L} + ln({M})) = {x}"
        )
    return x


def get_inverse_lms_pred_fev1_for_zscore(pred_fev1_arr, S, M, L):
    return (np.exp(L * np.log(pred_fev1_arr / M)) - 1) / (S * L)


def calc_predicted_value_LMS(spline_vals, coeffs, height: int, age: int, debug=False):
    """
    Implemented from the GLI reference equations.
    The GLI's model a location (M), scale (S), shape (L) model to predicted FEV1 given a Z-Score​
    Equations: https://www.ers-education.org/lr/show-details/?idP=138978
    Paper: https://www.ersnet.org/science-and-research/ongoing-clinical-research-collaborations/the-global-lung-function-initiative/gli-tools/
    """
    # M = exp(a0 + a1*ln(Height) + a2*ln(Age) + a3*AfrAm + a4*NEAsia + a5*SEAsia + Mspline)
    M = np.exp(
        coeffs["M"]["Intercept"]
        + coeffs["M"]["Height"] * np.log(height)
        + coeffs["M"]["Age"] * np.log(age)
        + spline_vals["Mspline"]
    )

    # S =  exp(p0 + p1*ln(Age) + p2*AfrAm + p3*NEAsia + p4*SEAsia + Sspline)
    S = np.exp(
        coeffs["S"]["Intercept"]
        + coeffs["S"]["Age"] * np.log(age)
        + spline_vals["Sspline"]
    )

    L = coeffs["L"]["Intercept"] + coeffs["L"]["Age"] * np.log(age)
    if debug:
        print(
            f"M = exp({coeffs['M']['Intercept']} + {coeffs['M']['Height']}*ln({height}) + {coeffs['M']['Age']}*ln({age}) + {spline_vals['Mspline']})"
        )
        print(
            f"S = exp({coeffs['S']['Intercept']} + {coeffs['S']['Age']}*ln({age}) + {spline_vals['Sspline']})"
        )
        print(f"L = {coeffs['L']['Intercept']} + {coeffs['L']['Age']}*ln({age})")

    # Get lower limit of normal (5th percentile)
    LLN = get_lms_pred_value_for_zscore(-1.644, M, S, L, debug)

    # The Z-score of a value indicates how far from the mean is that value, in units of standard deviation.
    # In the LMS model, percentile_value(zscore) = exp(ln(1 - z-score *L*S)/L +ln(M))
    # Hence, the standard deviation is: percentile_value(zscore)
    n_std = 1
    sigma_up = get_lms_pred_value_for_zscore(n_std, M, S, L) - M
    sigma_down = M - get_lms_pred_value_for_zscore(-n_std, M, S, L)

    return {
        "M": M,
        "sigma_up": sigma_up,
        "sigma_down": sigma_down,
        "LLN": LLN,
        "L": L,
        "S": S,
    }
