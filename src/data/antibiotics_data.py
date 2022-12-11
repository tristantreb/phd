import pandas as pd


def load(datadir):
    print("\n** Loading antibiotics data **")
    df = pd.read_excel(datadir + "clinicaldata_updated.xlsx", sheet_name="Antibiotics")
    n_initial_entries = df.shape[0]

    # Drop columns that are not needed
    # List of columns to keep
    tmp_columns = df.columns
    columns_to_keep = [
        "ID",
        "Antibiotic Name",
        "Route",
        "Home IV's'",
        "Start Date",
        "Stop Date",
    ]
    df = df[columns_to_keep]
    print("\n* Dropping unnecessary columns from antibiotics data *")
    print("Columns filetered: {}".format(columns_to_keep))
    print("Columns dropped: {}".format(set(tmp_columns) - set(columns_to_keep)))

    # Enfore data types
    df = df.astype({"ID": str, "Antibiotic Name": str, "Route": str, "Home IV's'": str})
    # Cast datetime to date
    df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.date
    df["Stop Date"] = pd.to_datetime(df["Stop Date"]).dt.date

    return df
