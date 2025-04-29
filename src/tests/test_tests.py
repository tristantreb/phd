import numpy as np
import pandas as pd


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