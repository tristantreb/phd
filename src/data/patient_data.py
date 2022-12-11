import pandas as pd


def load(datadir):
    patient_data = pd.read_excel(
        datadir + "clinicaldata_updated.xlsx", sheet_name="Patients", dtype={"ID": str}
    )
    print("** Loading patient data **")
    n_initial_entries = patient_data.shape[0]))

    # Drop columns that are not needed
    # List of columns to keep
    columns = patient_data.columns
    columns_to_keep = [
        "ID",
        "DOB",
        "Age",
        "Sex",
        "Height",
        "Weight",
        "Predicted FEV1",
        "FEV1 Set As",
    ]
    patient_data = patient_data[columns_to_keep]
    print("\n** Dropping unnecessary columns from patient data **")
    print("Filtering columns: {}".format(columns_to_keep))
    print("Columns dropped: {}".format(set(columns) - set(columns_to_keep)))

    # Clean data types
    patient_data.Weight = patient_data.Weight.replace(to_replace="75,4", value="75.4")

    # Enforce the data types
    patient_data = patient_data.astype(
        {
            "ID": str,
            "Age": int,
            "Sex": str,
            "Height": float,
            "Weight": float,
            "Predicted FEV1": float,
            "FEV1 Set As": float,
        }
    )
    patient_data.DOB = pd.to_datetime(patient_data.DOB).dt.date

    # Clean data
    patient_data = _correct_patient_data(patient_data)

    # Data sanity checks
    patient_data.apply(_age_sanity_check, axis=1)
    patient_data.apply(_sex_sanity_check, axis=1)
    patient_data.apply(_height_sanity_check, axis=1)
    patient_data.apply(_weight_sanity_check, axis=1)
    # Not added because need to decide which PRedicted FEV1 to use
    # patient_data.apply(_predicted_fev1_sanity_check, axis=1)
    # patient_data.apply(_fev1_set_as_sanity_check, axis=1)

    print("Loaded patient data with {} entries ({} initially)".format(patient_data.shape[0], n_initial_entries))
    return patient_data


def _age_sanity_check(row):
    assert (
        row.Age >= 18 and row.Age <= 70
    ), "Age ({}) outside 18-70 range for ID {}".format(row.Age, row.ID)
    return -1


def _sex_sanity_check(row):
    assert row.Sex in [
        "Male",
        "Female",
    ], "Sex ({}) is not 'Male' neither 'Female' for ID {}".format(row.Sex, row.ID)
    return -1

def _height_sanity_check(row):
    assert (
        row.Height >= 120 and row.Height <= 220
    ), "Height ({}) outside 120-220cm range for ID {}".format(row.Height, row.ID)
    return -1

def _weight_sanity_check(row):
    assert (
        row.Weight >= 30 and row.Weight <= 120
    ), "Weight ({}) outside 30-120kg range for ID {}".format(row.Weight, row.ID)
    return -1


def _correct_patient_data(patient_data):
    print("\n** Correcting patient data **")

    # ID 60, convert height from m to cm
    tmp = patient_data.loc[patient_data.ID == "60", "Height"]
    patient_data.Height.loc[patient_data.ID == "60"] = tmp * 100

    # Print data correction for ID 60
    print(
        "Corrected height for ID 60 from {} to {}".format(
            (tmp).to_string(index=False),
            (tmp * 100).to_string(index=False),
        )
    )
    # ID 66, convert height from m to cm
    tmp = patient_data.loc[patient_data.ID == "66", "Height"]
    patient_data.loc[patient_data.ID == "66", "Height"] = tmp * 100

    # Print data correction for ID 66
    print(
        "Corrected height for ID 66 from {} to {}".format(
            (tmp).to_string(index=False),
            (tmp * 100).to_string(index=False),
        )
    )
    return patient_data