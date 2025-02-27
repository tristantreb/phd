import numpy as np
import pandas as pd

import src.inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat


def get_mock_data():
    df_mock = pd.DataFrame(
        {
            "ID": ["1", "1", "1"],
            "Date Recorded": [1, 2, 3],
            "Height": 180,
            "Age": 35,
            "Sex": "Male",
            "ecFEV1": [1.8, 2.2, 2.2],
            "ecFEF2575%ecFEV1": [12, 120, 150],
            "idx ecFEV1 (L)": [1, 2, 2],
            "idx ecFEF2575%ecFEV1": [0, 6, 7],
            "idx ecFEF25-75 % ecFEV1 (%)": [0, 6, 7],
        }
    )
    df_mock["Date Recorded"] = pd.to_datetime(
        df_mock["Date Recorded"], unit="D", origin="2020-01-01"
    )
    return df_mock


def test_light_model_no_o2sat():
    df_mock = get_mock_data()
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_Gmain0.2_Gtails10_w0.73"
    ecfev1_noise_model_suffix = "_std0.7"
    (
        _,
        p_M_given_D,
        AR_given_M_and_D,
    ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        light=True,
        n_days_consec=3,
    )

    print(AR_given_M_and_D.shape)

    ar_day_1 = AR_given_M_and_D[0, :]

    return -1
