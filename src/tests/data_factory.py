import numpy as np
import pandas as pd

import models.builders as mb


def get_data_df_template(n_days):
    df_mock = pd.DataFrame(
        {
            "ID": [f"1" for i in range(n_days)],
            "Date Recorded": [i for i in range(n_days)],
            "Height": 180,
            "Age": 35,
            "Sex": "Male",
        }
    )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock


def get_mock_data(fev1_mode="changing"):
    if fev1_mode == "identical":
        df_mock = pd.DataFrame(
            {
                "ID": ["1", "1", "1"],
                "Date Recorded": [1, 2, 3],
                "Height": 180,
                "Age": 35,
                "Sex": "Male",
                "ecFEV1": [2.2, 2.2, 2.2],
                "ecFEF2575%ecFEV1": [97, 97, 97],
            }
        )
    elif fev1_mode == "changing":
        df_mock = pd.DataFrame(
            {
                "ID": ["1", "1", "1"],
                "Date Recorded": [1, 2, 3],
                "Height": 180,
                "Age": 35,
                "Sex": "Male",
                # VE against VE work
                "ecFEV1": [4.2, 2.2, 1.8],
                # "ecFEV1": [2.2, 2.2, 2.2],
                # "ecFEV1": [4.2, 2.2, 1.8],
                "ecFEF2575%ecFEV1": [90, 90, 90],
                # VE against VE fails - 3rd day shifts right
                # "ecFEV1": [2.2, 2.2, 4.2],
                # "ecFEF2575%ecFEV1": [90, 90, 90],
                # VE aganst VE fail - weird becasue almost right
                # "ecFEV1": [2.2, 2.2, 4.5],
                # "ecFEF2575%ecFEV1": [90, 90, 90],
                # "ecFEF2575%ecFEV1": [12, 120, 150],
            }
        )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock


def get_mock_data_2_days(fev1_mode):
    if fev1_mode == "identical":
        df_mock = pd.DataFrame(
            {
                "ID": ["1", "1"],
                "Date Recorded": [1, 2],
                "Height": 180,
                "Age": 35,
                "Sex": "Male",
                "ecFEV1": [1.8, 1.8],
                "ecFEF2575%ecFEV1": [50, 50],
            }
        )
    elif fev1_mode == "changing":
        df_mock = pd.DataFrame(
            {
                "ID": ["1", "1"],
                "Date Recorded": [1, 2],
                "Height": 180,
                "Age": 35,
                "Sex": "Male",
                "ecFEV1": [3.5, 5.5],
                "ecFEF2575%ecFEV1": [30, 50],
            }
        )

    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock


def add_idx_obs_cols(df, ecFEV1, ecFEF2575prctecFEV1=None):
    df["idx ecFEV1 (L)"] = [ecFEV1.get_bin_idx_for_value(x) for x in df["ecFEV1"]]
    # If there are no idx col, then ecFEF2575prctecFEV1 is not added to evidence in custom cutset cond alg
    if ecFEF2575prctecFEV1 is not None:
        df["idx ecFEF2575%ecFEV1"] = [
            ecFEF2575prctecFEV1.get_bin_idx_for_value(x) for x in df["ecFEF2575%ecFEV1"]
        ]
    else:
        df["idx ecFEF2575%ecFEV1"] = np.nan
    df["idx ecFEF25-75 % ecFEV1 (%)"] = df["idx ecFEF2575%ecFEV1"]
    return df


def get_df_with_no_obs(n_days):
    height, age, sex = 180, 35, "Male"
    df_mock = pd.DataFrame(
        {
            "ID": [f"1" for i in range(n_days)],
            "Date Recorded": [i for i in range(n_days)],
            "Height": height,
            "Age": age,
            "Sex": sex,
        }
    )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    (
        _,
        _,
        HFEV1,
        uFEV1,
        ecFEV1,
        AR,
        _,
        S,
    ) = mb.fev1_fef2575_long_model_noise_shared_healthy_vars_and_temporal_ar(
        height,
        age,
        sex,
        ar_change_cpt_suffix="_shape_factor_single_laplace_1.6",
        ecfev1_noise_model_suffix="_std_add_mult_ecfev1",
        fef2575_cpt_suffix="",
    )
    ecfev1 = 2
    df_mock.loc[0, "ecFEV1"] = ecfev1
    df_mock.loc[1:, "ecFEV1"] = np.nan
    # Idx must be an int
    df_mock["idx ecFEV1 (L)"] = np.array(
        [ecFEV1.get_bin_idx_for_value(ecfev1)] + [np.nan] * (n_days - 1), dtype=object
    )
    df_mock.loc[0, "ecFEF2575%ecFEV1"] = np.nan
    df_mock.loc[0, "idx ecFEF2575%ecFEV1"] = np.nan
    df_mock.loc[0, "idx ecFEF25-75 % ecFEV1 (%)"] = np.nan
    print(f"datafactroy idx ecFEV1 (L): {df_mock.loc[0, 'idx ecFEV1 (L)']}")

    return df_mock
