import logging

import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import data.sanity_checks as sanity_checks


def fetch_dataset(study_name):
    df = pd.read_csv(
        f"{dh.get_path_to_main()}/DataFiles/TR/tromso_28april2025.csv",
        delimiter=";",
        decimal=",",
    )

    df = df.filter(regex="T5")
    df = df.dropna(axis=0, how="all")

    # Add ID
    df.reset_index(inplace=True)
    df.rename(columns={"index": f"UID"}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": f"ID"}, inplace=True)
    return df


def to_breathe_colnames(df, study_name):

    df.columns = df.columns.map(
        lambda item: item.split(f"_{study_name}")[0].replace("_", " ").capitalize()
    )
    df = df.rename(
        columns={
            "Uid": "UID",
            "Id": "ID",
            "Respiratory infection 3w": "Respiratory infection 3W",
            "Fev1": "FEV1",
            "Fef25 75": "FEF2575",
            "Mean oxygen saturation": "O2 Saturation",
        },
    )
    return df


def format_age(x):
    try:
        return int(x)
    except:
        # if age is 80+, set to 80
        if x == "80+":
            return 80
        else:
            logging.warning(f"AGE: Error converting {x} to float")
            return None


def format_sex(x):
    if x == 1:
        return "Male"
    elif x == 0:
        return "Female"
    else:
        logging.warning(f"SEX: Error converting {x} to string")
        return None


def _correct_fev1(df):
    # df = dh.remove_recording(df, 8971, "FEV1", 0.145)
    # df = dh.remove_recording(df, 17547, "FEV1", 0.105)
    return df


def _correct_fef2575(df):
    return df


def process_tr_data(df, study_name):
    df.Age = df.Age.apply(format_age)
    df.Sex = df.Sex.apply(format_sex)

    df = _correct_fev1(df)
    df = _correct_fef2575(df)

    cols = ["Age", "Sex", "Height", "FEV1", "FEF2575", "O2 Saturation"]
    sanity_checks.data_types(df[cols])

    logging.info("Processing Age")
    df.apply(lambda x: sanity_checks.age(x["Age"], x["ID"], 80), axis=1)

    logging.info("Processing Sex")
    df.apply(lambda x: sanity_checks.sex(x["Sex"], x["ID"]), axis=1)

    logging.info("Processing Height")
    df.apply(lambda x: sanity_checks.height(x["Height"], x["ID"]), axis=1)

    logging.info("Processing FEV1")
    df.apply(lambda x: sanity_checks.fev1(x["FEV1"], x["ID"]), axis=1)

    logging.info("Processing FEF2575")
    df.apply(lambda x: sanity_checks.fef2575(x["FEF2575"], x["ID"]), axis=1)

    logging.info("Processing O2 Saturation")
    df.apply(lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x["ID"]), axis=1)

    # logging.info(f"{df.shape[0]} individuals in {study_name}")

    return df


def load_tromso_data(study_name):
    df = fetch_dataset(study_name)
    df = to_breathe_colnames(df, study_name)
    df = process_tr_data(df, study_name)
    return df


def build_meas_df(study_name):
    print("\n*** Building O2 Saturation and FEV1 dataframe ***")
    df = load_tromso_data(study_name)

    cols = ["Age", "Sex", "Height", "FEV1", "FEF2575", "O2 Saturation"]
    df = df.dropna(subset=cols).reset_index(drop=True)
    df = bd.calc_predicted_FEV1_LMS_df(df, 1, debug=False)
    df = bd.calc_healthy_O2_sat_df(df)
    df = bd.calc_FEV1_prct_predicted_df(df, with_ecFEV1=False)
    df = bd.calc_O2_sat_prct_healthy_df(df)
    return df
