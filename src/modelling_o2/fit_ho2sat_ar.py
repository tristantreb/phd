import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy.optimize import curve_fit, minimize


# Group by Airway Resistance and take 80th percentile of O2 Sat / Healthy O2 Sat if there are more than 50 observations
def calc_rmax_o2(df_for_AR, y, percentile=80):
    # return np.percentile(df_for_AR[y], percentile), len(df_for_AR)
    # Take data between 80 and 90th percentile
    return np.percentile(
        df_for_AR[y],
        range(percentile - 5, percentile + 5),
    ).mean(), len(df_for_AR)


def fit_factor_profile(df_to_fit, x, y):
    x_data = df_to_fit[x].values
    y_data = df_to_fit[y].values

    # Piecewise fit (constant + polynomial)
    def func(x, x0, y0, k1, k2, k3):
        # x0 = 43
        # y0 = df_to_fit[df_to_fit[x] < x0][
        #     o2_col
        # ].mean()

        return np.piecewise(
            x,
            [x <= x0],
            [
                lambda x: y0,
                lambda x: k1 * np.power((x - x0), 3)
                + k2 * np.power((x - x0), 2)
                + k3 * (x - x0)
                + y0,
            ],
        )

    def objective(params, x, y):
        return np.sum((func(x, *params) - y)**2)

    # Enforce monotonicity constraint
    # constraints = ({'type': 'ineq', 'fun': lambda params: np.diff(func(x_data, *params))})

    # Initial guess for parameters
    # initial_guess = [4.34232599e+01, 8.92599726e-01, -3.60069643e-04, 1.56798589e-02, -2.12605357e-01]

    # # Minimize the objective function with the constraint
    # result = minimize(objective, initial_guess, args=(x_data, y_data), constraints=constraints)
    # parameters = result.x

    parameters, covariance = curve_fit(
        func,
        df_to_fit[x].values,
        df_to_fit[y].values,
    )
    print(f"Parameters: {parameters}")
    df_to_fit["Piecewise fit"] = func(x_data, *parameters)

    # Spline fit
    ## Base value for smoothing parameter
    s = df_to_fit.shape[0] - np.sqrt(2 * df_to_fit.shape[0])
    print(f"Smoothing parameter: {s}")
    ### Create a spline representation of the curve
    ### tck-tuple: (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    tck = splrep(
        x_data,
        y_data,
        s=s,
    )
    ### Evalute the spline repr on a new set of points
    df_to_fit["Spline"] = BSpline(*tck)(df_to_fit[x])

    # Mean smoothing
    df_to_fit["Mean Smoothing"] = (
        df_to_fit[y].rolling(5, center=True).mean()
    )
    return df_to_fit
