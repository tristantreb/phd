import concurrent.futures

import numpy as np
import pandas as pd

import src.data.breathe_data as bd
import src.data.helpers as dh
import src.inference.inf_two_days_model as inf_two_days_model
import src.inference.long_inf_slicing as slicing
import src.models.builders as mb

df = bd.load_meas_from_excel("BR_O2_FEV1_FEF2575_conservative_smoothing_with_idx")


def process_id(id):
    df_for_ID = df[df["ID"] == id]
    res = inf_two_days_model.infer_for_id(df_for_ID, debug=False)
    return res


if __name__ == "__main__":
    ids = df["ID"].unique()

    # num_cores = os.cpu_count()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_id, ids))

    final_results = pd.concat(results, ignore_index=True)
    final_results.to_excel(
        f"{dh.get_path_to_main()}/ExcelFiles/BR/Refining_F3/infer_AR_using_two_days_model_ecFEV1_ecfev1noiseaddmult.xlsx"
    )
