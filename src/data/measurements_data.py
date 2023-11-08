import pandas as pd
import sanity_checks

datadir = "../../../../SmartCareData/"


def load():
    df = load_measurements_without_smartcare_id()
    n_initial_entries = df.shape[0]

    df_id_map = load_id_map()
    df = merge_with_id_mapping(df, df_id_map)

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


def _correct_patient_id(df, id, true_patient_id):
    print(
        "Correct ID {}'s Patient_ID from {} to {}".format(
            id, df.Patient_ID[df.SmartCareID == id].values, true_patient_id
        )
    )
    df.loc[df["SmartCareID"] == id, "Patient_ID"] = true_patient_id
    return df


def load_id_map(with_correction=True):
    # The ID map files links each SmartCareID to a Patient_ID and a Study_ID
    # The SmartCare ID is the identifier used in the SmartCare clinical data
    # The Patient_ID is the identifier used in the measurements data
    # The Study_ID is the identifier used in ??? - No idea I've never needed it

    print("\n** Loading ID map **")

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


def merge_with_id_mapping(
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
        # "FEV 1 %",  # Is this FEV 1 / Predicted FEV? Compare this to our calculated version of Predicted FEV1 %
        "Weight in Kg",
        "O2 Saturation",
        "Pulse (BPM)",
        "Rating",  # What is this?
        "Temp (deg C)",
        # "Activity - Steps",
        # "Activity - Points",
    ]
    df = df[columns_to_keep]
    print("\n* Dropping unnecessary columns from measurements data *")
    print("Columns filtered {}".format(columns_to_keep))
    print("Dropping columns {}".format(set(tmp_columns) - set(columns_to_keep)))

    # Rename columns
    print("\n* Renaming columns *")
    rename_dic = {
        "Date/Time recorded": "Date recorded",
        "FEV 1": "FEV1",
        # "FEV 1 %": "FEV1 %",
        "Weight in Kg": "Weight (kg)",
    }
    df.rename(
        columns=rename_dic,
        inplace=True,
    )
    print("Renamed columns {}".format(rename_dic))

    # Enforce data types
    df = df.astype(
        {
            "User ID": str,
            "UserName": str,
            "Recording Type": str,
            "FEV1": float,
            # "FEV1 %": float,
            "Weight (kg)": float,
            "O2 Saturation": float,
            "Pulse (BPM)": int,
            "Rating": int,
            "Temp (deg C)": float,
            # "Activity - Steps": int,
            # "Activity - Points": int,
        },
        # This allows to ignore errors when casting. We added this because you can't cast a NaN to int
        errors="ignore",
    )
    # Cast datetime to date
    df["Date recorded"] = pd.to_datetime(df["Date recorded"]).dt.date

    # Apply data sanity checks
    print("\n* Applying data sanity checks *")

    print("\nFEV1")
    df = _correct_fev1(df)
    df.apply(lambda x: sanity_checks.fev1(x.FEV1, x["UserName"]), axis=1)

    print("\nWeight (kg)")
    df = _correct_weight(df)
    df.apply(lambda x: sanity_checks.weight(x["Weight (kg)"], x["UserName"]), axis=1)

    print("\nPulse (BPM)")
    df = _correct_pulse(df)
    df.apply(lambda x: sanity_checks.pusle(x["Pusle (BPM)"], x["UserName"]), axis=1)

    print("\nO2 Saturation")
    df = _correct_O2_saturation(df)
    df.apply(
        lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x["UserName"]), axis=1
    )

    print("\nTemp (deg C)")
    df = _correct_temp(df)
    df.apply(
        lambda x: sanity_checks.temperature(x["Temp (deg C)"], x["UserName"]), axis=1
    )

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


def _remove_recording(df, username, column, value):
    idx = df[(df["UserName"] == username) & (df[column] == value)].index
    print(
        "Dropping {} entries with {} = {} for user {}".format(
            idx.shape[0], column, value, username
        )
    )
    df.drop(idx, inplace=True)
    return df


def _correct_fev1(df):
    # ID 54 has a series of FEV1 values < 2, no values for 6 months, then a value of 3.5
    # This 3.45 value is possible, but it'll give more noise than meaning to our model - especially until we introduce time
    # Let's remove it
    df = _remove_recording(df, "Kings004", "FEV1", 3.45)
    return df


def _correct_O2_saturation(df):
    # Find all rows where O2 Saturation is outside 70-100 % range
    idx = df[(df["O2 Saturation"] < 70) | (df["O2 Saturation"] > 100)].index
    # Print unique IDs with number of rows with O2 Saturation outside 70-100 % range
    print(
        "IDs with O2 Saturation outside 70-100 % range: \n{}".format(
            df.loc[idx, ["UserName", "O2 Saturation"]].sort_values("UserName")
        )
    )
    # Remove rows with O2 Saturation outside 70-100 % range
    print(
        "Dropping {} entries with O2 Saturation outside 70-100 % range".format(len(idx))
    )
    return df.drop(idx, inplace=False)


def _correct_weight(df):
    # Warning - ID PapworthSummer has Weight (28.9375) outside 30-122 kg range
    # Warning - ID PapworthSummer has Weight (29.200000000000003) outside 30-122 kg range
    # Warning - ID EmemTest has Weight (14.9625) outside 30-122 kg range
    # Warning - ID FPH0011 has Weight (7.8) outside 30-122 kg range
    # PapworthSummer, EmemTest and FPH0011 are test users that are not in the id mapping file.
    # They'd be removed with merge_with_id_mapping(), so we can ignore them here.

    # Warning - ID Papworth033 has Weight (6.0) outside 30-122 kg range
    # Warning - ID Papworth033 has Weight (6.0) outside 30-122 kg range
    # This is SmartCare ID 134, which has a weight of 61 kg in the clinical data.
    # Let's remove those 2 entries
    df = _remove_recording(df, "Papworth033", "Weight (kg)", 6.0)

    # Warning - ID Kings013 has Weight (0.55) outside 30-122 kg range
    # This is SmartCare ID 63, which has a weight of 54.3 kg in the clinical data.
    # Let's remove this entry
    df = _remove_recording(df, "Kings013", "Weight (kg)", 0.55)

    # Warning - ID Papworth017 has Weight (8.262500000000001) outside 30-122 kg range
    # This is SmartCare ID 102, which has a wieght of 57.8 kg in the clinical data.
    # Let's remove this entry
    df = _remove_recording(df, "Papworth017", "Weight (kg)", 8.262500000000001)

    # Warning - ID leeds01730 has Weight (1056.0) outside 30-122 kg range
    # This is SmartCare ID 70, which has a weight of 105.2 kg in the clinical data.
    # Let's remove this entry
    df = _remove_recording(df, "leeds01730", "Weight (kg)", 1056.0)

    # Warning - ID Papworth019 has Weight (20.0) outside 30-122 kg range
    # This is SmartCare ID 113, which has a weight of 50.4 kg in the clinical data.
    # Let's remove this entry
    df = _remove_recording(df, "Papworth019", "Weight (kg)", 20.0)
    return df


def _correct_pulse(df):
    # For some reason, there are some entries with Pulse (BPM) == 511, we'll remove them
    # Drop idx where Pusle (BPM) is equal to 511
    idx = df[df["Pulse (BPM)"] == 511].index
    print("Dropping {} entries with Pulse (BPM) == 511)".format(idx.shape[0]))
    print(df.loc[idx, ["Pulse (BPM)", "UserName"]])
    df.drop(idx, inplace=True)

    # Warning - UserName leeds01670 has Pulse (30.0) outside 40-200 range
    # This recording is clearly an outlier, all other recordings for this user are above 80 BPM
    # Let's remove this entry
    df = _remove_recording(df, "leeds01670", "Pulse (BPM)", 30.0)

    return df


def _correct_temp(df):
    # Drop idx where Temp (deg C) is below 15
    idx = df[df["Temp (deg C)"] < 15].index
    print("Dropping {} entries with Temp (deg C) < 15".format(idx.shape[0]))
    # Print all temp values and usernames
    print(df.loc[idx, ["Temp (deg C)", "UserName"]])
    df.drop(idx, inplace=True)

    # Drop idx where Temp (deg C) is above 60
    idx = df[df["Temp (deg C)"] > 60].index
    print("Dropping {} entries with Temp (deg C) > 60".format(idx.shape[0]))
    print(df.loc[idx, ["Temp (deg C)", "UserName"]])
    df.drop(idx, inplace=True)
    return df
