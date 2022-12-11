import pandas as pd


def load(datadir):
    print("\n** Loading measurements data **")
    df = pd.read_csv(datadir + "mydata.csv")
    n_initial_entries = df.shape[0]

    # Get ID (same as SmartCare ID)
    # ID-patient_hash map
    id_map = pd.read_excel(datadir + "patientidnew.xlsx")

    # Merge
    print("\n* Merging measurements data with ID map to retrieve SmartCare ID *")
    # List User IDs from measurements data that don't have a corresponding Patient_ID in ID map
    user_id_list = df[~df["User ID"].isin(id_map["Patient_ID"])]["User ID"].unique()
    print("List User IDs that have no SmartCare ID {}".format(user_id_list))
    # Print number of rows with User ID from measurements data not present in Patient_ID from ID map
    print(
        "{} rows with User ID from measurements data not present in Patient_ID from ID map".format(
            df[~df["User ID"].isin(id_map["Patient_ID"])].shape[0]
        )
    )
    # Generate fake SmartCare IDs for those User IDs
    print(
        "\nGenerating fake SmartCare IDs for those User IDs, SmartCare IDs are within 23-241 range, we'll generate fake IDs starting 300."
    )
    id_map = id_map.append(
        pd.DataFrame(
            {
                "Patient_ID": user_id_list,
                "SmartCareID": [300 + i for i in range(len(user_id_list))],
            }
        )
    )

    df = df.merge(
        id_map,
        how="left",
        left_on="User ID",
        right_on="Patient_ID",
        copy=True,
        validate="many_to_one",
    )

    # Print SmartCare IDs sorted
    print("All IDs sorted: {}".format(df["SmartCareID"].sort_values().unique()))

    print(
        "{} entries left after merge (initial {}, removed {})".format(
            df.shape[0], n_initial_entries, n_initial_entries - df.shape[0]
        )
    )

    # Drop columns that are not needed
    # List of columns to keep
    tmp_columns = df.columns
    columns_to_keep = [
        "SmartCareID",
        "Recording Type",
        "Date/Time recorded",
        "FEV 1",
        "Predicted FEV",  # What is this?
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
            "SmartCareID": "ID",
            "Date/Time recorded": "Date recorded",
            "FEV 1": "FEV1",
            "FEV 1 %": "FEV1 %",
            "Weight in Kg": "Weight",
            "Temp (deg C)": "Temperature",
        },
        inplace=True,
    )

    # Enforce data types
    df = df.astype(
        {
            "ID": str,
            "Recording Type": str,
            "FEV1": float,
            "Predicted FEV": float,
            "FEV1 %": float,
            # "Weight": float,
            "O2 Saturation": float,
            # "Pulse (BPM)": int,
            # "Rating": int,
            # "Temperature": float,
            # "Activity - Steps": int,
            # "Activity - Points": int,
        }
    )
    # Cast datetime to date
    df["Date recorded"] = pd.to_datetime(df["Date recorded"]).dt.date

    # Correct erroneous data
    # print("\n* Correcting measurements data *")
    # df = _correct_measurements_data(df)

    # Apply data sanity checks
    print("\n* Applying data sanity checks *")
    df.apply(_fev1_sanity_check, axis=1)
    # df["Predicted FEV"] = df.apply(_predicted_fev_sanity_check, axis=1)
    df = _O2_saturation_sanity_check(df)

    # Look for duplicates
    print("\n* Looking for duplicates *")
    duplicates = df.duplicated(
        subset=["ID", "Recording Type", "Date recorded"], keep="last"
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
            "Warning - ID {} has FEV1 ({}) outside 0.1-5.5 L range".format(
                row.ID, row.FEV1
            )
        )
    return -1


# def _predicted_fev_sanity_check(row):
#     if row["Predicted FEV"] < 1 or row["Predicted FEV"] > 5.5:
#         print("Warning - ID {} has Predicted FEV ({}) outside 1-5.5 L range".format(row.ID, row["Predicted FEV"]))
#     return -1


def _O2_saturation_sanity_check(df):
    print("O2 Saturation sanity check")
    # Find all rows where O2 Saturation is outside 70-100 % range
    idx = df[(df["O2 Saturation"] < 70) | (df["O2 Saturation"] > 100)].index
    # Print unique IDs with number of rows with O2 Saturation outside 70-100 % range
    print(
        "IDs with O2 Saturation outside 70-100 % range: \n{}".format(
            df.loc[idx, ["ID", "O2 Saturation"]].sort_values("ID")
        )
    )
    # Remove rows with O2 Saturation outside 70-100 % range
    print("Removing {} rows with O2 Saturation outside 70-100 % range".format(len(idx)))
    df.drop(idx, inplace=True)
    return df
