import numpy as np

import src.inference.long_inf_slicing as slicing
import src.models.builders as mb


def sample_jointly_from_AR(df_two_days, date_1, date_2, light_model=False):
    df_two_days = df_two_days.copy().reset_index(drop=True)
    height = df_two_days.loc[0, "Height"]
    age = df_two_days.loc[0, "Age"]
    sex = df_two_days.loc[0, "Sex"]
    # id = df_two_days.loc[0, "ID"]
    (
        _,
        inf_alg,
        HFEV1,
        uecFEV1,
        ecFEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        ecFEF2575prctecFEV1,
    ) = mb.o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars(
        height, age, sex
    )
    if light_model:
        (
            _,
            inf_alg,
            HFEV1,
            uecFEV1,
            ecFEV1,
            AR,
            HO2Sat,
            O2SatFFA,
            IA,
            UO2Sat,
            O2Sat,
            ecFEF2575prctecFEV1,
        ) = mb.o2sat_fev1_fef2575_point_in_time_model_noise_shared_healthy_vars_light(
            height, age, sex
        )

    # Set variables parametrisation
    key_hfev1 = f"['{uecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
    key_ho2sat = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {HO2Sat.name}"
    HFEV1.set_factor_node_key(key_hfev1)
    HO2Sat.set_factor_node_key(key_ho2sat)

    # 1/ Infer AR1 using the two consecutive days model
    df_res_final_epoch1, _, _ = (
        slicing.query_back_and_forth_across_days_specific_evidence(
            df_two_days,
            inf_alg,
            [HFEV1, HO2Sat],
            [AR],
            [ecFEV1.name, ecFEF2575prctecFEV1.name],
            1e-8,
            days_specific_evidence=[],
            max_passes=5,
        )
    )

    df_res_final_epoch1.set_index("Day", inplace=True)

    # 2/ Sample from AR1
    ar_day1_dist = df_res_final_epoch1.loc[date_1, AR.name]
    [ar_day1_sample] = AR.sample(n=1, p=ar_day1_dist)
    idx_ar = AR.get_bin_for_value(ar_day1_sample)[1]

    v = np.zeros(len(df_two_days)) - 1
    v[0] = ar_day1_sample
    df_two_days["AR"] = v

    v = np.zeros(len(df_two_days)) + 10000
    v[0] = idx_ar
    v = v.astype(int)
    df_two_days[f"idx {AR.name}"] = v

    # 3/ Infer AR2 using with sampled AR1 as evidence specific to day 1
    days_specific_evidence = [(AR.name, [date_1])]

    df_res_final_epoch2, _, _ = (
        slicing.query_back_and_forth_across_days_specific_evidence(
            df_two_days,
            inf_alg,
            [HFEV1, HO2Sat],
            [AR],
            [ecFEV1.name, ecFEF2575prctecFEV1.name],
            1e-8,
            days_specific_evidence,
            max_passes=5,
            debug=False,
        )
    )
    df_res_final_epoch2.set_index("Day", inplace=True)
    ar_day2_dist = df_res_final_epoch2.loc[date_2, AR.name]

    # Print the interquartile ranges of the AR distributions
    # ar1_1 = AR.get_val_at_quantile(ar_day1_dist, 0.25)
    # ar1_2 = AR.get_val_at_quantile(ar_day1_dist, 0.75)
    # print(f"AR1: {ar1_2} - {ar1_1} = {ar1_2 - ar1_1}")

    # ar2_1 = AR.get_val_at_quantile(ar_day2_dist, 0.25)
    # ar2_2 = AR.get_val_at_quantile(ar_day2_dist, 0.75)
    # print(f"AR2: {ar2_2} - {ar2_1} = {ar2_2 - ar2_1}")

    # 5/ Sample from AR2
    [ar_day2_sample] = AR.sample(n=1, p=ar_day2_dist)

    ar_shift = ar_day2_sample - ar_day1_sample

    return ar_shift
