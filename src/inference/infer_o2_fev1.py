# Based on the model in 2024-01-25_inference_on_FEV1_O2_model.ipynb
# Update 2025-05-01:
# 1/ noise model,
# 2/ data from july 2024,

import concurrent.futures

import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import models.builders as mb

# Checked that obs indices are correct, see ipynb mentioned above(01.05.2025)
df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


# Infer AR and IA for all data points given an individuals' age, sex, height, FEV1 and O2 saturation measurements
def infer_AR_IA_for_ID(df):
    df.reset_index(inplace=True)
    (
        _,
        inf_alg,
        HFEV1,
        uFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        _,
        IA,
        _,
        O2Sat,
        ecFEF2575prctecFEV1,
    ) = mb.o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars(
        df.Height[0],
        df.Age[0],
        df.Sex[0],
        ecfev1_noise_model_cpt_suffix="_std_add_mult_ecfev1",
        ar_prior="uniform",
        ia_prior="uniform",
        ar_fef2575_cpt_suffix="_ecfev1_2_days_model_add_mult_noise",
        check_model=False,
    )

    def infer_and_unpack(
        id,
        date_recorded,
        ecfev1_obs_idx,
        # id, date_recorded, ecfev1_obs_idx, o2sat_obs_idx, ecfef2575prctecfev1_obs_idx
    ):
        res = inf_alg.query(
            variables=[AR.name, HFEV1.name, HO2Sat.name],
            # variables=[AR.name, IA.name, HFEV1.name, HO2Sat.name],
            evidence={
                ecFEV1.name: ecfev1_obs_idx,
                # O2Sat.name: o2sat_obs_idx,
                # ecFEF2575prctecFEV1.name: ecfef2575prctecfev1_obs_idx,
            },
        )
        return (
            id,
            date_recorded,
            res[AR.name].values,
            # res[IA.name].values,
            res[HFEV1.name].values,
            res[HO2Sat.name].values,
        )

    res = df.apply(
        lambda row: infer_and_unpack(
            row["ID"],
            row["Date Recorded"],
            row[f"idx {ecFEV1.name}"],
            # row[f"idx {O2Sat.name}"],
            # row[f"idx {ecFEF2575prctecFEV1.name}"],
        ),
        axis=1,
    )
    return res


def process_id(id):
    df_for_ID = df[df.ID == id]
    res = infer_AR_IA_for_ID(df_for_ID)
    return res


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = df.ID.unique()
        res = executor.map(process_id, ids)

    res = pd.concat(res, ignore_index=True)
    res = (
        res.apply(pd.Series)
        .reset_index(drop=True)
        .rename(
            columns={
                0: "ID",
                1: "Date Recorded",
                2: "AR",
                # 3: "IA",
                3: "HFEV1",
                4: "HO2Sat",
            }
        )
    )

    res.to_excel(
        f"{dh.get_path_to_main()}/ExcelFiles/BR/infer_AR_using_fev1_01052025.xlsx",
        # f"{dh.get_path_to_main()}/ExcelFiles/BR/infer_AR_using_o2sat_fev1_01052025.xlsx",
        index=False,
    )
