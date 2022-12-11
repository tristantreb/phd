import pandas as pd

from .measurements_data import *
from .patient_data import *


def create_O2_FEV1_df(datadir):
    # patient_ID-patient_hash map
    id_map = (
        pd.read_excel(datadir + "patientidnew.xlsx", dtype={"SmartCareID": str})
        .drop("Study_ID", axis=1)
        .rename(columns={"SmartCareID": "ID"})
    )

    # Load measurements data

    # Load clinical data
    # Patient data: Information describing the patient
    patient_data = patient_data.load(datadir)

    # Load antibiotics data, cast datetime to date
    antibioticsdata = pd.read_excel(
        datadir + "clinicaldata_updated.xlsx",
        sheet_name="Antibiotics",
        dtype={"ID": str},
    )
    antibioticsdata["Start Date"] = pd.to_datetime(
        antibioticsdata["Start Date"]
    ).dt.date
    antibioticsdata["Stop Date"] = pd.to_datetime(antibioticsdata["Stop Date"]).dt.date

    # # Get all O2 and FEV1 measurements and merge them
    # O2 = extract_measure(measurements, "O2 Saturation")
    # FEV1 = extract_measure(measurements, "FEV1")
    # len_outer_join = O2.merge(FEV1, on=["ID", "Date recorded"], how="outer").shape[0]
    # O2_FEV1 = O2.merge(FEV1, on=["ID", "Date recorded"], how="inner")
    # print(
    #     "Removed {} rows with O2_FEV1 inner join, kept {:.0%} of measurements ({})".format(
    #         len_outer_join - O2_FEV1.shape[0],
    #         O2_FEV1.shape[0] / len_outer_join,
    #         O2_FEV1.shape[0],
    #     )
    # )

    # # Remove duplicates
    # # TODO: take last or average? How to best handle duplicates?
    # # Remove duplicated measurements
    # len_all_fev1_o2 = O2_FEV1.shape[0]
    # O2_FEV1 = O2_FEV1[
    #     O2_FEV1.duplicated(subset=["ID", "Date recorded"], keep="last") == False
    # ]
    # print(
    #     "Removed {} duplicates, {} measurements left".format(
    #         len_all_fev1_o2 - O2_FEV1.shape[0], O2_FEV1.shape[0]
    #     )
    # )

    return -1


# Function to extract one column from the data
# TODO: check that there's only one measurement per day
def extract_measure(measurements_in, label):
    # Could also filter by Recording Type
    measurements_out = measurements_in[measurements_in[label].notnull()][
        ["ID", "Date recorded", label]
    ]
    print("{} contains {} measurements".format(label, measurements_out.shape[0]))
    return measurements_out
