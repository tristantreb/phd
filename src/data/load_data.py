import measurements_data
import pandas as pd
import patient_data


def create_O2_FEV1_df(datadir):

    # Load measurements data
    measurements_data = measurements_data.load(datadir)

    # Load clinical data
    # Patient data
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
