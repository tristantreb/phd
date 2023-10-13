import numpy as np
import pandas as pd

PATH_LMS_EQ_LOOKUP_TABLES = (
    "../../../../DataFiles/PredictedFEV1/FromGLI/GLI_LMS_equations_lookuptables.xls"
)


def calc_predicted_fev1(height: int, age: int, sex: str):
    """
    Calculate predicted FEV1 according to the formula given by the lung function people at Royal Papworth Hospital
    This formula takes Age in years, Height in m and Sex
    Input Height is in cm, hence the /100
    """

    if sex == "Male":
        pred_FEV1 = (height / 100) * 4.3 - age * 0.029 - 2.49
        std_dev = 0.4
        return {"Predicted FEV1": pred_FEV1, "std": std_dev}
    elif sex == "Female":
        pred_FEV1 = (height / 100) * 3.95 - age * 0.025 - 2.6
        std_dev = 0.35
        return {"Predicted FEV1": pred_FEV1, "std": std_dev}


def calc_LMS_predicted_FEV1(spline_vals, coeffs, height: int, age: int, sex: str):
    """
    Implemented from the GLI reference equations.
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

    # Get lower limit of normal (5th percentile)
    LLN = np.exp(np.log(1 - 1.645 * L * S) / L + np.log(M))

    return {"Predicted FEV1": M, "std": S, "LLN": LLN}


def load_LMS_spline_vals(age: int, sex: str):
    """
    Get the spline values for the M and S curves from the lookup table
    We ignore the Lspline column as it is always 0 for males or females
    """
    if sex == "Male":
        sheet_name = "FEV1 males"
    elif sex == "Female":
        sheet_name = "FEV1 females"
    else:
        raise ValueError(f"Sex {sex} not in Male/Female")

    df = pd.read_excel(
        PATH_LMS_EQ_LOOKUP_TABLES,
        sheet_name=sheet_name,
        header=1,
        usecols="B:E",
    )
    Mspline = df[df.age == age].Mspline.values[0]
    Sspline = df[df.age == age].Sspline.values[0]
    return {"Mspline": Mspline, "Sspline": Sspline}


def load_LMS_coeffs(sex: str):
    """
    Get the coefficients for the L, M and S curves from the lookup table
    """
    if sex == "Male":
        sheet_name = "FEV1 males"
    elif sex == "Female":
        sheet_name = "FEV1 females"
    else:
        raise ValueError(f"Sex {sex} not in Male/Female")

    df = pd.read_excel(
        PATH_LMS_EQ_LOOKUP_TABLES,
        sheet_name=sheet_name,
        header=2,
        usecols="G:I, K, L, N, O",
        index_col=0,
        nrows=7,
        names=["Coeff", "M_coeff", "M_val", "S_coeff", "S_val", "L_coeff", "L_val"],
    )
    return {
        "M": {
            "Intercept": df.loc["Intercept", "M_val"],
            "Height": df.loc["Height", "M_val"],
            "Age": df.loc["Age", "M_val"],
        },
        "S": {"Intercept": df.loc["Intercept", "S_val"], "Age": df.loc["Age", "S_val"]},
        "L": {"Intercept": df.loc["Intercept", "L_val"], "Age": df.loc["Age", "L_val"]},
    }
