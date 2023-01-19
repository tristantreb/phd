import biology
import pandas as pd

datadir = "../../../../SmartCareData/"


def load():
    df = load_measurements_without_smartcare_id()
    n_initial_entries = df.shape[0]

    df_id_map = load_id_map()
    df = get_smartcare_id(df, df_id_map)

    # Drop columns "User ID" and "UserName" as they are not needed
    df.drop(columns=["User ID", "UserName", "Study_ID"], inplace=True)

    # Drop rows with NaN in "SmartCareID" as they are not needed
    df.dropna(subset=["ID"], inplace=True)

    # Move ID to first column
    df.insert(0, "ID", df.pop("ID"))

    print(
        "\n{} entries left after merge (initial {}, removed {})".format(
            df.shape[0],
            n_initial_entries,
            n_initial_entries - df.shape[0],
        )
    )
    return df


def load_id_map(with_correction=True):
    # The ID map files links each SmartCareID to a Patient_ID and a Study_ID
    # The SmartCare ID is the identifier used in the SmartCare clinical data
    # The Patient_ID is the identifier used in the measurements data
    # The Study_ID is the identifier used in ??? - No idea I've never needed it

    df_id_map = pd.read_excel(datadir + "patientidnew.xlsx")

    # Enforce data types
    df_id_map = df_id_map.astype(
        {"Patient_ID": str, "Study_ID": str, "SmartCareID": str}
    )

    # Correct SmartCare IDs
    if with_correction:
        df_id_map = _correct_patient_id(df_id_map, "101", "0HeWh64M_zc5U5l2xqzAs4")
        df_id_map = _correct_patient_id(df_id_map, "125", "1au5biSTt0bNWgfI0WItr5")
        df_id_map = _correct_patient_id(df_id_map, "232", "-TKpptiCA5cASNKU0VSmx4")
        df_id_map = _correct_patient_id(df_id_map, "169", "-Cujq-NEcld_Keu_W1-Nw5")
        df_id_map = _correct_patient_id(df_id_map, "38", "-Q0Wf614z94DSTy6nXjyw7")

    # Replace SmartCareID with ID
    df_id_map.rename(columns={"SmartCareID": "ID"}, inplace=True)

    return df_id_map


def _correct_patient_id(df, id, true_patient_id):
    print(
        "Correct ID {}'s Patient_ID from {} to {}".format(
            id, df.Patient_ID[df.SmartCareID == id].values, true_patient_id
        )
    )
    df.loc[df["SmartCareID"] == id, "Patient_ID"] = true_patient_id
    return df


def get_smartcare_id(
    df_measurements, df_id_map, left_merge_on="User ID", right_merge_on="Patient_ID"
):
    # Merge
    print("\n* Merging measurements data with ID map to retrieve SmartCare ID *")
    # List User IDs from measurements data that don't have a corresponding Patient_ID in ID map
    user_id_list = (
        df_measurements[~df_measurements["User ID"].isin(df_id_map["Patient_ID"])]
        .apply(
            lambda x: "username {}, user id {}".format(x.UserName, x["User ID"]), axis=1
        )
        .unique()
    )

    print("List User IDs that have no SmartCare ID\n{}".format(user_id_list))

    df_measurements = df_measurements.merge(
        df_id_map,
        how="left",
        left_on=left_merge_on,
        right_on=right_merge_on,
        copy=True,
        validate="many_to_one",
    )
    return df_measurements


def load_measurements_without_smartcare_id():
    print("\n** Loading measurements data **")
    df = pd.read_csv(datadir + "mydata.csv")
    n_initial_entries = df.shape[0]

    # Drop columns that are not needed
    # List of columns to keep
    tmp_columns = df.columns
    columns_to_keep = [
        "User ID",
        "UserName",
        "Recording Type",
        "Date/Time recorded",
        "FEV 1",
        # "Predicted FEV",  # We use the calc version
        "FEV 1 %",  # Is this FEV 1 / Predicted FEV?
        "Weight in Kg",
        "O2 Saturation",
        "Pulse (BPM)",
        "Rating",  # What is this?
        "Temp (deg C)",
        "Activity - Steps",
        "Activity - Points",
    ]
    df = df[columns_to_keep]
    print("\n* Dropping unnecessary columns from measurements data *")
    print("Columns filtered {}".format(columns_to_keep))
    print("Dropping columns {}".format(set(tmp_columns) - set(columns_to_keep)))

    # Rename columns
    df.rename(
        columns={
            "Date/Time recorded": "Date recorded",
            "FEV 1": "FEV1",
            "FEV 1 %": "FEV1 %",
            "Weight in Kg": "Weight (kg)",
        },
        inplace=True,
    )

    # Enforce data types
    df = df.astype(
        {
            "User ID": str,
            "UserName": str,
            "Recording Type": str,
            "FEV1": float,
            # "Predicted FEV": float,
            "FEV1 %": float,
            "Weight (kg)": float,
            "O2 Saturation": float,
            "Pulse (BPM)": int,
            "Rating": int,
            "Temp (deg C)": float,
            "Activity - Steps": int,
            "Activity - Points": int,
        },
        # This allows to ignore errors when casting. We added this because you can't cast a NaN to int
        errors="ignore",
    )
    # Cast datetime to date
    df["Date recorded"] = pd.to_datetime(df["Date recorded"]).dt.date

    # Correct erroneous data
    # print("\n* Correcting measurements data *")
    # df = _correct_measurements_data(df)

    # Apply data sanity checks
    print("\n* Applying data sanity checks *")
    df.apply(_fev1_sanity_check, axis=1)
    df = _O2_saturation_sanity_check(df)

    # Look for duplicates
    print("\n* Looking for duplicates *")
    duplicates = df.duplicated(
        subset=["UserName", "Recording Type", "Date recorded"], keep="last"
    )
    print(
        "Found {} duplicates, saving them in DataFiles/SmartCare/duplicates.xlsx".format(
            duplicates.sum()
        )
    )
    # Save all duplicates in an excel file
    # df[duplicates].to_excel(datadir + "../DataFiles/SmartCare/duplicates.xlsx")
    # Remove duplicates from df
    print("Removing {} duplicated entries".format(duplicates.sum()))
    df = df[~duplicates]

    print(
        "\nLoaded measurements data with {} entries (initially {}, removed {})".format(
            df.shape[0], n_initial_entries, n_initial_entries - df.shape[0]
        )
    )
    return df


def _fev1_sanity_check(row):
    if row.FEV1 < 0.1 or row.FEV1 > 5.5:
        print(
            "Warning - UserName {} has FEV1 ({}) outside 0.1-5.5 L range".format(
                row.UserName, row.FEV1
            )
        )
    return -1


def _O2_saturation_sanity_check(df):
    print("O2 Saturation sanity check")
    # Find all rows where O2 Saturation is outside 70-100 % range
    idx = df[(df["O2 Saturation"] < 70) | (df["O2 Saturation"] > 100)].index
    # Print unique IDs with number of rows with O2 Saturation outside 70-100 % range
    print(
        "IDs with O2 Saturation outside 70-100 % range: \n{}".format(
            df.loc[idx, ["UserName", "O2 Saturation"]].sort_values("UserName")
        )
    )
    # Remove rows with O2 Saturation outside 70-100 % range
    print("Removing {} rows with O2 Saturation outside 70-100 % range".format(len(idx)))
    df.drop(idx, inplace=True)
    return df
