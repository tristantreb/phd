import pandas as pd


def get_mock_data():
    df_mock = pd.DataFrame(
        {
            "ID": ["1", "1", "1"],
            "Date Recorded": [1, 2, 3],
            "Height": 180,
            "Age": 35,
            "Sex": "Male",
            "ecFEV1": [1.8, 2.2, 2.2],
            "ecFEF2575%ecFEV1": [12, 120, 150],
            "idx ecFEV1 (L)": [1, 2, 2],
            "idx ecFEF2575%ecFEV1": [0, 6, 7],
            "idx ecFEF25-75 % ecFEV1 (%)": [0, 6, 7],
        }
    )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock
