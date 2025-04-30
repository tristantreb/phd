import logging

import pandas as pd

import data.helpers as dh
import data.sanity_checks as sanity_checks

df = pd.read_csv(
    f"{dh.get_path_to_main()}/DataFiles/Tromso/tromso_28april2025.csv",
    delimiter=";",
    decimal=",",
)

df.reset_index(inplace=True)
df.rename(columns={"index": "ID"}, inplace=True)


# Split datasets
def get_dataset(df, study_name):
    df = df.filter(regex=study_name)
    df = df.dropna(axis=0, how="all")
    df.reset_index(inplace=True)
    df.rename(columns={"index": f"ID {study_name}"}, inplace=True)
    logging.info(f"{df.shape[0]} individuals in {study_name}")
    return df


df_t5 = get_dataset(df, "T5")
df_t6 = get_dataset(df, "T6")
df_t7 = get_dataset(df, "T7")

# Assert sum of rows is equal to df
# assert (
#     df_t5.shape[0] + df_t6.shape[0] + df_t7.shape[0] == df.shape[0]
# ), logging.error("Sum of datasets size is not equal to main dataset size")


# Format age
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


df_t5["AGE_T5"] = df_t5["AGE_T5"].apply(format_age)


# Format sex
def format_sex(x):
    if x == 1:
        return "Male"
    elif x == 0:
        return "Female"
    else:
        logging.warning(f"SEX: Error converting {x} to string")
        return None


df_t5["SEX_T5"] = df_t5["SEX_T5"].apply(format_sex)


# No need to format height, FEV1, FEF25_75, O2 sat


# Map to colnames from breathe dataset
def tromso_to_breathe_colnames():
    return {
        "AGE_T5": "Age",
        "SEX_T5": "Sex",
        "HEIGHT_T5": "Height",
        "FEV1_T52": "FEV1",
        "FEF25_75_T52": "FEF2575",
        "MEAN_OXYGEN_SATURATION_T52": "O2 Saturation",
    }


df_t5.rename(columns=tromso_to_breathe_colnames(), inplace=True)

# Check data types for breathe like columns
cols = ["Age", "Sex", "Height", "FEV1", "FEF2575", "O2 Saturation"]
sanity_checks.data_types(df_t5[cols])

# Apply sanity checks
logging.info("Processing Age")
df_t5.apply(lambda x: sanity_checks.age(x["Age"], x["ID"]), axis=1)

logging.info("Processing Sex")
df_t5.apply(lambda x: sanity_checks.sex(x["Sex"], x["ID"]), axis=1)

logging.info("Processing Height")
df_t5.apply(lambda x: sanity_checks.height(x["Height"], x["ID"]), axis=1)

logging.info("Processing FEV1")
df_t5.apply(lambda x: sanity_checks.fev1(x["FEV1"], x["ID"]), axis=1)

logging.info("Processing FEF2575")
df_t5.apply(lambda x: sanity_checks.fef2575(x["FEF2575"], x["ID"]), axis=1)

logging.info("Processing O2 Saturation")
df_t5.apply(lambda x: sanity_checks.o2_saturation(x["O2 Saturation"], x["ID"]), axis=1)
