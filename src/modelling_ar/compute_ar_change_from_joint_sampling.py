"""
As part of 2024-07-29_AR_change_factor
Script written to parallelize the computation across IDs
"""

import concurrent.futures

import numpy as np
import pandas as pd

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.modelling_ar.AR_joint_sampling as model_ar

df_obs = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def get_ar_shift_with_joint_sampling_for_ID(df_for_ID, max_offset=3):
    df_for_ID = df_for_ID.reset_index(drop=True)
    id = df_for_ID.loc[0, "ID"]
    print(f"Processing ID {id} with {len(df_for_ID)} entries")

    res = pd.DataFrame()
    for n_idx_offset in list(np.arange(1, max_offset + 1)):
        print(f"ID: {id}, offset: {n_idx_offset}")

        for i, row in df_for_ID.iterrows():
            # If the offset is too large, break
            if i + n_idx_offset >= len(df_for_ID):
                print(f"ID: {id}, idx: {i}, offset: {n_idx_offset}, breaking")
                break
            # print(f"ID: {id}, idx: {i}, offset: {n_idx_offset}")
            # Find idx of max ecFEV1
            idx_max_ecFEV1 = idx_max_FEV1 = df_for_ID.sort_values(
                # by=["ecFEV1", "ecFEF2575", "O2 Saturation"], ascending=False
                by=["ecFEV1", "ecFEF2575%ecFEV1"],
                ascending=False,
            ).index[0]
            # Get two first days as well as idx_max_ecFEV1
            idx_two_days = [i, i + n_idx_offset]

            # If the max ecFEV1 is not in the two days, add it to have more accurate results
            if idx_max_ecFEV1 not in idx_two_days:
                # df_two_days = df_for_ID.iloc[idx_two_days]
                # Check that IQR reduces when adding the max ecFEV1: use ID 134
                df_two_days = df_for_ID.iloc[
                    idx_two_days + [idx_max_ecFEV1.item()]
                ].reset_index(drop=True)
            else:
                df_two_days = df_for_ID.iloc[idx_two_days].reset_index(drop=True)

            day_1 = df_two_days.loc[0, "Date Recorded"]
            day_2 = df_two_days.loc[1, "Date Recorded"]

            ar_shift = model_ar.sample_jointly_from_AR(
                df_two_days, day_1, day_2, debug=False
            )
            days_elapsed = (day_2 - day_1).days

            # Add row to table with format: ID, date, days elapsed, AR shift, offset
            new_row = pd.DataFrame(
                data=[
                    [
                        df_two_days.loc[0, "ID"],
                        df_two_days.loc[0, "Date Recorded"],
                        days_elapsed,
                        ar_shift,
                        n_idx_offset,
                    ]
                ],
                columns=[
                    "ID",
                    "Date Recorded",
                    "Days elapsed",
                    "AR samples shift",
                    "Offset",
                ],
            )
            res = pd.concat([res, new_row])
        return res


def process_id(id):
    print(f"Processing ID {id}")
    df_for_ID = df_obs[df_obs.ID == id].reset_index(drop=True)
    res = get_ar_shift_with_joint_sampling_for_ID(df_for_ID)
    return res


if __name__ == "__main__":

    with concurrent.futures.ProcessPoolExecutor() as executor:
        ids = df_obs.ID.unique()
        results = executor.map(process_id, ids)

    final_result = pd.concat(results, ignore_index=True)
    final_result.to_excel(
        f"{dh.get_path_to_main()}/ExcelFiles/BR/Refining_F3/AR_joint_samples_diff_for_change_factor_ecfev1noisestd0.23.xlsx",
        index=False,
    )
