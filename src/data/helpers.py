import pandas as pd


def compute_avg(df, col_name, unit):
    """
    Compute avg of col_name per individual
    """
    tmp = df.groupby("ID")[col_name].mean()
    # Add tmp to a new column per Id
    df = df.join(tmp, on="ID", rsuffix="_avg")

    df[f"ID (avg {col_name})"] = df.apply(
        lambda x: f"{x.ID} ({str(round(x[f'{col_name}_avg'],1))}{unit})",
        axis=1,
    )
    return df
