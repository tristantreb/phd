import antibiotics_data
import measurements_data
import pandas as pd
import patient_data

datadir = "../../../../SmartCareData/"


def create():
    # Load measurements data
    df_measurements = measurements_data.load()

    # Load clinical data
    # Patient data
    df_patient = patient_data.load()

    # Load antibiotics data, cast datetime to date
    df_antibiotics = antibiotics_data.load()

    print("\n** Creating DataFrame for O2 FEV1 analysis **")
    # Extract O2 and FEV1 measurements
    df_O2 = extract_measure(df_measurements, "O2 Saturation")
    df_FEV1 = extract_measure(df_measurements, "FEV1")

    # Merge O2 with FEV1
    df_O2_FEV1 = pd.merge(df_O2, df_FEV1, on=["ID", "Date recorded"], how="outer")
    n_na = df_O2_FEV1.isna().sum().sum()
    df_O2_FEV1.dropna(inplace=True)
    print(
        "Merged O2 and FEV1 into {} entries (initially {}, removed {} NaN)".format(
            df_O2_FEV1.shape[0], df_O2_FEV1.shape[0] + n_na, n_na
        )
    )

    # Assert that there's only one set of measurements per ID per day
    print("Asserting that there's only one measurement per ID per day")
    assert (
        df_O2_FEV1.groupby(["ID", "Date recorded"]).size().max() == 1
    ), "There's more than one measurement per day for some patients"

    # Merge O2_FEV1 with patient data
    df_O2_FEV1 = pd.merge(df_O2_FEV1, df_patient, on="ID", how="left")

    # # Merge O2_FEV1 with antibiotics data
    # df_O2_FEV1 = pd.merge(df_O2_FEV1, df_antibiotics, on="ID", how="outer")

    print(
        "\nCreated df_O2_FEV1 with {} entries (initially {}, removed {})".format(
            df_O2_FEV1.shape[0], df_O2_FEV1.shape[0] + n_na, n_na
        )
    )
    return df_O2_FEV1


# Function to extract one column from the data
# TODO: check that there's only one measurement per day
def extract_measure(measurements_in, label):
    # Could also filter by Recording Type
    measurements_out = measurements_in[measurements_in[label].notnull()][
        ["ID", "Date recorded", label]
    ]
    print("{} has {} measurements".format(label, measurements_out.shape[0]))
    return measurements_out
