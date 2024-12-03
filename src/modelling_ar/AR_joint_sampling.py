import numpy as np

import src.inference.long_inf_slicing as slicing
import src.models.builders as mb


def sample_jointly_from_AR(df_two_days, date_1, date_2, light_model=False, debug=False):
    df_two_days = df_two_days.copy().reset_index(drop=True)
    height = df_two_days.loc[0, "Height"]
    age = df_two_days.loc[0, "Age"]
    sex = df_two_days.loc[0, "Sex"]
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
        height,
        age,
        sex,
        ecfev1_noise_model_cpt_suffix="_std_0.068",
        ar_fef2575_cpt_suffix="_ecfev1_2_days_model_add_mult_noise",
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

    # Since O2 sat side is unobserved, we can set a uniform precomputed messaged for it
    arr = np.ones(AR.card)
    arr /= arr.sum()
    uniform_from_o2_side = {
        "['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)'] -> Airway resistance (%)": arr
    }

    # 1/ Infer AR1 using the two consecutive days model
    df_res_final_epoch1, _, _ = slicing.query_forwardly_across_days(
        # slicing.query_back_and_forth_across_days_specific_evidence(
        df_two_days,
        inf_alg,
        [HFEV1, HO2Sat],
        [AR],
        [ecFEV1.name, ecFEF2575prctecFEV1.name],
        1e-8,
        days_specific_evidence=[],
        precomp_messages=uniform_from_o2_side.copy(),
        debug=debug,
    )

    df_res_final_epoch1.set_index("Day", inplace=True)

    # 2/ Randomly sample from AR1
    date_1_str = date_1.strftime("%Y-%m-%d")
    ar_day1_dist = df_res_final_epoch1.loc[date_1_str, AR.name]
    [ar_day1_sample] = AR.sample(n=1, p=ar_day1_dist)

    # Add sample to df
    df_two_days[AR.name] = -1.1
    df_two_days[f"idx {AR.name}"] = 1000
    df_two_days.loc[0, AR.name] = ar_day1_sample
    df_two_days.loc[0, f"idx {AR.name}"] = AR.get_bin_idx_for_value(ar_day1_sample)

    # 3/ Infer AR2 using with sampled AR1 as evidence specific to day 1
    days_specific_evidence = [(AR.name, [date_1])]

    df_res_final_epoch2, _, _ = slicing.query_forwardly_across_days(
        # slicing.query_back_and_forth_across_days_specific_evidence(
        df_two_days,
        inf_alg,
        [HFEV1, HO2Sat],
        [AR],
        [ecFEV1.name, ecFEF2575prctecFEV1.name],
        1e-8,
        days_specific_evidence,
        precomp_messages=uniform_from_o2_side.copy(),
        debug=debug,
    )
    df_res_final_epoch2.set_index("Day", inplace=True)
    date_2_str = date_2.strftime("%Y-%m-%d")
    ar_day2_dist = df_res_final_epoch2.loc[date_2_str, AR.name]

    # 5/ Sample from AR2
    [ar_day2_sample] = AR.sample(n=1, p=ar_day2_dist)

    if debug:
        print(
            f"Day 1 ecFEV1 obs: {df_two_days.loc[0, 'ecFEV1']}, ecFEF2575 obs: {df_two_days.loc[0, 'ecFEF2575%ecFEV1']}, AR: {ar_day1_sample}, AR dist: {ar_day1_dist}"
        )
        print(
            f"Day 2 ecFEV1 obs: {df_two_days.loc[1, 'ecFEV1']}, ecFEF2575 obs: {df_two_days.loc[1, 'ecFEF2575%ecFEV1']}, AR: {ar_day2_sample}, AR dist {ar_day2_dist}"
        )
    ar_shift = ar_day2_sample - ar_day1_sample

    return ar_shift
