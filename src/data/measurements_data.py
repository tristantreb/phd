import pandas as pd


def load(datadir):
    df = pd.read_csv(datadir + "mydata.csv").rename(columns={"FEV 1": "FEV1"})
    # Cast datetime to date
    df["Date recorded"] = pd.to_datetime(
        df["Date/Time recorded"]
    ).dt.date
    # Get ID (same as SmartCare ID)
    df = df.merge(
        id_map, left_on="User ID", right_on="Patient_ID", copy=True
    )
    df.drop(columns=["User ID", "UserName", "Patient_ID"], inplace=True)
