import pandas as pd

import src.data.measurements_data as measurements_data
import src.data.patient_data as patient_data
import src.modelling_fev1.pred_fev1 as pred_fev1

datadir = "../../../../SmartCareData/"


def create():
    df_measurements = measurements_data.load()
    df_patient = patient_data.load()

    print("\n** Creating DataFrame for O2 FEV1 analysis **")
    # Extract O2 and FEV1 measurements
    df_O2 = extract_measure(df_measurements, "O2 Saturation")
    df_FEV1 = extract_measure(df_measurements, "FEV1")

    # Merge O2 with FEV1
    df_O2_FEV1 = pd.merge(df_O2, df_FEV1, on=["ID", "Date Recorded"], how="outer")
    n_na = df_O2_FEV1.isna().sum().sum()

    # Dropping all NaNs, because the df contains only O2 and FEV1 measurements
    df_O2_FEV1.dropna(inplace=True)
    print(
        "Merged O2 and FEV1 into {} entries (initially {}, removed {} NaN)".format(
            df_O2_FEV1.shape[0], df_O2_FEV1.shape[0] + n_na, n_na
        )
    )

    # Assert that there's only one set of measurements per ID per day
    print("Asserting that there's only one measurement per ID per day")
    assert (
        df_O2_FEV1.groupby(["ID", "Date Recorded"]).size().max() == 1
    ), "There's more than one measurement per day for some patients"

    # Merge O2_FEV1 with patient data
    df_O2_FEV1 = pd.merge(df_O2_FEV1, df_patient, on="ID", how="left")
    df_O2_FEV1.sort_values(["ID", "Date Recorded"], inplace=True)

    # Compute FEV1 % Predicted
    df_O2_FEV1 = pred_fev1.calc_FEV1_prct_predicted_df(df_O2_FEV1)

    print(
        "\nCreated df_O2_FEV1 with {} entries (initially {}, removed {})".format(
            df_O2_FEV1.shape[0], df_O2_FEV1.shape[0] + n_na, n_na
        )
    )
    return df_O2_FEV1


# Function to extract one column from the data
# TODO: check that there's only one measurement per day
def extract_measure(measurements_in, label, with_patient_id=False):
    # Could also filter by Recording Type
    if with_patient_id:
        measurements_out = measurements_in[measurements_in[label].notnull()][
            ["User ID", "Date Recorded", label]
        ]
    else:
        measurements_out = measurements_in[measurements_in[label].notnull()][
            ["ID", "Date Recorded", label]
        ]
    print("{} has {} measurements".format(label, measurements_out.shape[0]))
    return measurements_out
