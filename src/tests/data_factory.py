import pandas as pd


def get_mock_data(light=False):
    df_mock = pd.DataFrame(
        {
            "ID": ["1", "1", "1"],
            "Date Recorded": [1, 2, 3],
            "Height": 180,
            "Age": 35,
            "Sex": "Male",
            "ecFEV1": [1.8, 2.2, 4.2],
            "ecFEF2575%ecFEV1": [12, 120, 150],
        }
    )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock


def add_idx_obs_cols(df, ecFEV1, ecFEF2575prctecFEV1):
    df["idx ecFEV1 (L)"] = [ecFEV1.get_bin_idx_for_value(x) for x in df["ecFEV1"]]
    df["idx ecFEF2575%ecFEV1"] = [
        ecFEF2575prctecFEV1.get_bin_idx_for_value(x) for x in df["ecFEF2575%ecFEV1"]
    ]
    df["idx ecFEF25-75 % ecFEV1 (%)"] = df["idx ecFEF2575%ecFEV1"]
    return df
