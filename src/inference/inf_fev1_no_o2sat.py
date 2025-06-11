import concurrent.futures

import numpy as np
import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import inference.long_inf_slicing as slicing
import models.builders as mb

# Checked that obs indices are correct, see ipynb mentioned above(01.05.2025)
df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def infer_for_id(df_for_ID, debug, diff_threshold=1e-8):
    """
    General function to infer values on the 2 days model for a given ID.
    With precomputed messages

    Adapt before using
    """
    df_for_ID = df_for_ID.reset_index(drop=True)
    if debug:
        print(f"\nID: {df_for_ID.ID.iloc[0]}")
        print(f"#datapoints: {len(df_for_ID)}")

    height = df_for_ID.Height.iloc[0]
    age = df_for_ID.Age.iloc[0]
    sex = df_for_ID.Sex.iloc[0]

    ecfev1_noise_model_cpt_suffix = "_std_add_mult_ecfev1"
    ar_fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"
    (
        _,
        inf_alg,
        HFEV1,
        uFEV1,
        ecFEV1,
        AR,
        ecFEF2575prctecFEV1,
    ) = mb.fev1_fef2575_point_in_time_model_noise_shared_healthy_vars(
        height,
        age,
        sex,
        ecfev1_noise_model_cpt_suffix=ecfev1_noise_model_cpt_suffix,
        ar_fef2575_cpt_suffix=ar_fef2575_cpt_suffix,
    )

    vars = [AR]
    shared_vars = [HFEV1]
    obs_vars = [ecFEV1.name]
    # obs_vars = [ecFEV1.name, ecFEF2575prctecFEV1.name]

    # Find the max FEV1 values
    # Given an ID, get the data which maximises ecFEV1, then ecFEF2575, then O2 Saturation
    idx_max_FEV1 = df_for_ID.sort_values(
        by=["ecFEV1", "ecFEF2575"], ascending=False
    ).index[0]

    # For each entry, create a two_days data structure that hold the current day as well as the day where the max FEV1 is observed
    # If the two idx are the same, then run a one day model.
    # Adding the max FEV1 information to the model input allows a better estimation of the HFEV1, hereby reducing the shared uncertainty between AR and HFEV1.

    # Save information into a df
    res_for_ID = pd.DataFrame({})

    # Get precompupted messages to speedup inference
    arr = np.ones(AR.card)
    arr /= arr.sum()
    # Create precomp messages for FEF25-75 given it's unobserved
    # arr = np.ones(ecFEF2575prctecFEV1.card)
    # arr /= arr.sum()
    # uniform_from_fef2575_side = {
    #     "['ecFEF2575%ecFEV1', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    # }

    for index, row in df_for_ID.iterrows():
        ecfev1_obs_idx = row[f"idx {ecFEV1.name}"]
        # ecfef2575prctecfev1_obs_idx = row[f"idx {ecFEF2575prctecFEV1.name}"]
        id = row["ID"]

        res = inf_alg.query(
            variables=[AR.name, HFEV1.name],
            evidence={
                ecFEV1.name: ecfev1_obs_idx,
                # ecFEF2575prctecFEV1.name: ecfef2575prctecfev1_obs_idx,
            },
        )

        new_row = pd.DataFrame(
            {
                "ID": [id],
                "Date Recorded": [row["Date Recorded"]],
                AR.name: [res[AR.name].values],
                HFEV1.name: [res[HFEV1.name].values],
            }
        )
        res_for_ID = pd.concat([res_for_ID, new_row], ignore_index=True)
    return res_for_ID


def process_id(id):
    df_for_ID = df[df.ID == id]
    res = infer_for_id(df_for_ID, debug=False)
    return res


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = df.ID.unique()
        res = executor.map(process_id, ids)

    res = pd.concat(res, ignore_index=True)
    res.to_excel(
        f"{dh.get_path_to_main()}/ExcelFiles/BR/infer_AR_using_fev1_06062025_draft.xlsx",
        index=False,
    )
