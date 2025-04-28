import numpy as np
import pandas as pd


def get_mock_data(fev1_mode):
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
                # "ecFEV1": [4.2, 2.2, 1.8],
                # "ecFEF2575%ecFEV1": [90, 90, 90],
                # VE against VE fails - 3rd day shifts right
                # "ecFEV1": [2.2, 2.2, 4.2],
                # "ecFEF2575%ecFEV1": [90, 90, 90],
                # VE aganst VE fail - weird becasue almost right
                "ecFEV1": [2.2, 2.2, 4.5],
                "ecFEF2575%ecFEV1": [90, 90, 90],
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
    if ecFEF2575prctecFEV1 is not None:
        df["idx ecFEF2575%ecFEV1"] = [
            ecFEF2575prctecFEV1.get_bin_idx_for_value(x) for x in df["ecFEF2575%ecFEV1"]
        ]
    else:
        df["idx ecFEF2575%ecFEV1"] = np.nan
    df["idx ecFEF25-75 % ecFEV1 (%)"] = df["idx ecFEF2575%ecFEV1"]
    return df
