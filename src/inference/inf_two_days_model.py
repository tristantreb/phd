import concurrent.futures

import numpy as np
import pandas as pd

import data.breathe_data as bd
import data.helpers as dh
import inference.long_inf_slicing as slicing
import models.builders as mb

# Checked that obs indices are correct, see ipynb mentioned above(01.05.2025)
# df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")
df = dh.load_excel(
    f"{dh.get_path_to_main()}/ExcelFiles/SC/O2_FEV1_df_conservative_smoothing_with_idx.xlsx",
    date_cols=["Date Recorded"],
)


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
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    ) = mb.o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars(
        height,
        age,
        sex,
        ecfev1_noise_model_cpt_suffix=ecfev1_noise_model_cpt_suffix,
        ar_fef2575_cpt_suffix=ar_fef2575_cpt_suffix,
    )

    vars = [AR]
    shared_vars = [HFEV1, HO2Sat]
    # obs_vars = [ecFEV1.name]
    obs_vars = [ecFEV1.name, O2Sat.name]
    # obs_vars = [ecFEV1.name, ecFEF2575prctecFEV1.name]
    # obs_vars = [ecFEV1.name, O2Sat.name, ecFEF2575prctecFEV1.name]

    # Find the max FEV1 values
    # Given an ID, get the data which maximises ecFEV1, then ecFEF2575, then O2 Saturation
    idx_max_FEV1 = df_for_ID.sort_values(
        by=["ecFEV1", "O2 Saturation"], ascending=False
    ).index[0]
    # idx_max_FEV1 = df_for_ID.sort_values(
    #     by=["ecFEV1", "ecFEF2575", "O2 Saturation"], ascending=False
    # ).index[0]

    # For each entry, create a two_days data structure that hold the current day as well as the day where the max FEV1 is observed
    # If the two idx are the same, then run a one day model.
    # Adding the max FEV1 information to the model input allows a better estimation of the HFEV1, hereby reducing the shared uncertainty between AR and HFEV1.

    # Save information into a df
    df_current_day_res = pd.DataFrame({})

    # Get precompupted messages to speedup inference
    arr = np.ones(AR.card)
    arr /= arr.sum()
    uniform_from_o2_side = {
        "['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }
    # Create precomp messages for FEF25-75 given it's unobserved
    # arr = np.ones(ecFEF2575prctecFEV1.card)
    # arr /= arr.sum()
    # uniform_from_fef2575_side = {
    #     "['ecFEF2575%ecFEV1', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    # }

    for i, _ in df_for_ID.iterrows():
        if i != idx_max_FEV1:
            df_two_days = df_for_ID.iloc[[i, idx_max_FEV1]]
        else:
            df_two_days = df_for_ID.iloc[[i]]

        df_query_res_two_days, _, _ = slicing.query_forwardly_across_days(
            df_two_days,
            inf_alg,
            shared_vars,
            vars,
            obs_vars,
            diff_threshold,
            [],
            # precomp_messages=uniform_from_o2_side.copy(),
            # precomp_messages=uniform_from_o2_side.copy()
            # | uniform_from_fef2575_side.copy(),
            debug=debug,
        )

        new_row = df_query_res_two_days.loc[
            # 0, ["ID", "Day", HFEV1.name, HO2Sat.name, AR.name, IA.name]
            0,
            ["ID", "Day", HFEV1.name, HO2Sat.name, AR.name],
        ]
        df_current_day_res = pd.concat([df_current_day_res, pd.DataFrame(new_row).T])

    return df_current_day_res


def process_id(id):
    print(f"Processing ID: {id}")
    df_for_ID = df[df.ID == id]
    res = infer_for_id(df_for_ID, debug=False)
    return res


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = df.ID.unique()
        res = executor.map(process_id, ids)

    res = pd.concat(res, ignore_index=True)

    res.to_excel(
        f"{dh.get_path_to_main()}/ExcelFiles/SC/infer_AR_using_two_days_model_o2_fev1_06052025.xlsx",
        index=False,
    )
