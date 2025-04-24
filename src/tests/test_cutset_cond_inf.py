import numpy as np
from pgmpy.inference.ExactInference import VariableElimination
from plotly.subplots import make_subplots

import src.inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat
import src.inference.helpers as ih
import src.models.builders as mb
import src.tests.data_factory as data


def assert_low_element_wise_max_diff(v1, v2, tol=1e-10):
    assert np.max(np.abs(v1 - v2)) < tol


def test_3_days_model_with_cutset_cond_give_same_posteriors_as_var_elim():
    """
    End to end test for light model with only fev1 side
    """
    # Data
    df_mock = data.get_mock_data()
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_Gmain0.2_Gtails10_w0.73"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    # Load variable eliminiation
    (
        model,
        HFEV1,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_days_model_noise_shared_healthy_vars_and_temporal_ar(
        n_days,
        height,
        age,
        sex,
        ar_prior,
        ar_change_cpt_suffix,
        ecfev1_noise_model_suffix,
        light=False,
    )
    df_mock = data.add_idx_obs_cols(
        df_mock, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
    )
    var_elim = VariableElimination(model)

    # Run variable elimination
    evidence_dict = {}
    for i in range(n_days):
        evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]
        evidence_dict[ecFEF2575prctecFEV1_vars[i].name] = df_mock.loc[
            i, "idx ecFEF2575%ecFEV1"
        ]

    res_ve = var_elim.query(
        variables=[AR_vars[0].name, AR_vars[1].name, AR_vars[2].name, HFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    hfev1_ve = res_ve[HFEV1.name].values
    ar0_ve = res_ve[AR_vars[0].name].values
    ar1_ve = res_ve[AR_vars[1].name].values
    ar2_ve = res_ve[AR_vars[2].name].values

    # Run custom model
    (
        fig,
        p_M_given_D,
        p_HFEV1_given_D,
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        light=False,
        n_days_consec=3,
    )
    hfev1_cc = p_HFEV1_given_D
    ar0_cc = AR_given_M_and_D[0, :]
    ar1_cc = AR_given_M_and_D[1, :]
    ar2_cc = AR_given_M_and_D[2, :]

    AR = AR_vars[0]
    AR.name = AR.name.split(" day")[0]

    # plot_diff(
    #     HFEV1,
    #     AR_vars[0],
    #     hfev1_cc,
    #     hfev1_ve,
    #     ar0_cc,
    #     ar0_ve,
    #     ar1_cc,
    #     ar1_ve,
    #     ar2_cc,
    #     ar2_ve,
    # )

    assert_low_element_wise_max_diff(hfev1_cc, hfev1_ve)
    assert_low_element_wise_max_diff(ar0_cc, ar0_ve)
    assert_low_element_wise_max_diff(ar1_cc, ar1_ve)
    assert_low_element_wise_max_diff(ar2_cc, ar2_ve)

    return None


def plot_diff(
    HFEV1, AR, hfev1_cc, hfev1_ve, ar0_cc, ar0_ve, ar1_cc, ar1_ve, ar2_cc, ar2_ve
):
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.13)
    # Add HFEV1
    ih.plot_histogram(fig, HFEV1, hfev1_ve, HFEV1.a, HFEV1.b, 1, 1, annot=False)
    ih.plot_histogram(fig, AR, ar0_ve, AR.a, AR.b, 2, 1, annot=False)
    ih.plot_histogram(fig, AR, ar1_ve, AR.a, AR.b, 3, 1, annot=False)
    ih.plot_histogram(fig, AR, ar2_ve, AR.a, AR.b, 4, 1, annot=False)

    # Add HFEV1
    ih.plot_histogram(fig, HFEV1, hfev1_cc, HFEV1.a, HFEV1.b, 1, 1, title=HFEV1.name)
    ih.plot_histogram(
        fig,
        AR,
        ar0_cc,
        AR.a,
        AR.b,
        2,
        1,
        title=f"{AR.name} day 1",
        annot=False,
    )
    ih.plot_histogram(
        fig,
        AR,
        ar1_cc,
        AR.a,
        AR.b,
        3,
        1,
        title=f"{AR.name} day 2",
        annot=False,
    )
    ih.plot_histogram(
        fig,
        AR,
        ar2_cc,
        AR.a,
        AR.b,
        4,
        1,
        title=f"{AR.name} day 3",
        annot=False,
    )

    for i in range(4):
        fig.data[i].marker.color = "#636EFA"
        fig.data[i + 4].marker.color = "#EF553B"
    # Reduce x axis title font size
    fig.update_xaxes(title_font=dict(size=12), title_standoff=7)

    # Hide legend
    title = "Cutset conditioning (red) vs variable elimination (blue)"
    fig.update_layout(showlegend=False, height=550, width=800, title=title)
    fig.show()


def test_3_days_model_with_cutset_cond_provide_same_p_s_given_d_as_var_elim():
    df_mock = data.get_mock_data()
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_Gmain0.2_Gtails10_w0.73"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    (
        model,
        HFEV1,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
        S,
    ) = mb.fev1_fef2575_n_days_model_noise_shared_healthy_vars_and_temporal_ar_learn_S(
        n_days,
        height,
        age,
        sex,
        ar_prior,
        ar_change_cpt_suffix,
        ecfev1_noise_model_suffix,
        light=False,
    )
    var_elim = VariableElimination(model)

    df_mock = data.add_idx_obs_cols(
        df_mock, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
    )
    evidence_dict = {}
    for i in range(n_days):
        evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]
        evidence_dict[ecFEF2575prctecFEV1_vars[i].name] = df_mock.loc[
            i, "idx ecFEF2575%ecFEV1"
        ]

    res_ve_S = var_elim.query(
        variables=[S.name],
        evidence=evidence_dict,
        joint=False,
    )
    s_ve = res_ve_S[S.name].values

    # Run custom algorithm
    n_days_consec = 3
    (
        log_p_S_given_D,
        _,
    ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        n_days_consec=n_days_consec,
        light=False,
        get_p_s_given_d=True,
    )

    p_S_given_D = np.exp(log_p_S_given_D)
    p_S_given_D /= np.sum(p_S_given_D)

    # Assert results are equal
    assert_low_element_wise_max_diff(p_S_given_D, s_ve)


def test_2_days_model_with_cutset_cond_with_identical_and_changing_fev1():
    """
    Test that the custom algorithm provides different P(D|S) for identical and changing fev1 over two days
    Use two S: one that is identity (simulating one single AR across time), one that is changing (allowing different ARs)

    - P(D=same fev1|S=identity) > P(D=changing fev1|S=identity)
    - P(D=changing fev1|S=broad) > P(D=changing fev1|S=identity)
    - P(D|S=identity) must never be equal to P(D|S=broad)
    """
    df_mock_same_fev1 = data.get_mock_data_2_days(fev1_mode="identical")
    df_mock_changing_fev1 = data.get_mock_data_2_days(fev1_mode="changing")
    n_days = df_mock_same_fev1.shape[0]
    height, age, sex = df_mock_same_fev1.iloc[0][["Height", "Age", "Sex"]]

    # Model parameters
    ar_prior = "uniform"
    s_identity_cpt_suffix = "_shape_factor_identity"
    s_broad_cpt_suffix = "_shape_factor_Gmain0.2_Gtails10_w0.73"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    (
        model,
        HFEV1,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
        S,
    ) = mb.fev1_fef2575_n_days_model_noise_shared_healthy_vars_and_temporal_ar_learn_S(
        n_days,
        height,
        age,
        sex,
        ar_prior,
        s_broad_cpt_suffix,
        ecfev1_noise_model_suffix,
        light=False,
    )

    df_mock_same_fev1 = data.add_idx_obs_cols(
        df_mock_same_fev1, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
    )
    df_mock_changing_fev1 = data.add_idx_obs_cols(
        df_mock_changing_fev1, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
    )

    settings = {
        "si": (df_mock_same_fev1, s_identity_cpt_suffix),
        "sb": (df_mock_same_fev1, s_broad_cpt_suffix),
        "ci": (df_mock_changing_fev1, s_identity_cpt_suffix),
        "cb": (df_mock_changing_fev1, s_broad_cpt_suffix),
    }
    log_p_S_given_D = {}

    # Run custom algorithm
    n_days_consec = 3
    for setting_id in ["si", "sb", "ci", "cb"]:
        df_mock_tmp, s_cpt_suffix_tmp = settings[setting_id]
        (
            [log_p_S_given_D_tmp],
            _,
        ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
            df_mock_tmp,
            ar_prior=ar_prior,
            ar_change_cpt_suffix=s_cpt_suffix_tmp,
            ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
            n_days_consec=n_days_consec,
            light=False,
            get_p_s_given_d=True,
        )
        print(setting_id, log_p_S_given_D_tmp)
        log_p_S_given_D[setting_id] = log_p_S_given_D_tmp

    # Assert
    # Identity AR change factor favours when FEV1 is the same across time
    assert log_p_S_given_D["si"] > log_p_S_given_D["ci"]
    # Step change in FEV1 favours a board AR change factor
    assert log_p_S_given_D["ci"] < log_p_S_given_D["cb"]
    assert log_p_S_given_D["si"] != log_p_S_given_D["sb"]


def test_using_identity_AR_change_factor_against_single_AR_across_time_points_gives_same_results():
    """
    Test that using an identity AR change factor is equivalent to using a single AR across time points
    """
    # Data
    df_mock = data.get_mock_data()
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_identity"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    # Load variable eliminiation
    (
        model,
        HFEV1,
        AR,
        uecFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_days_model_noise_shared_healthy_vars_and_shared_ar(
        n_days,
        height,
        age,
        sex,
        ar_prior,
        ar_change_cpt_suffix,
        ecfev1_noise_model_suffix,
        light=False,
    )
    df_mock = data.add_idx_obs_cols(
        df_mock, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
    )
    var_elim = VariableElimination(model)

    # Run variable elimination
    evidence_dict = {}
    for i in range(n_days):
        evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]
        evidence_dict[ecFEF2575prctecFEV1_vars[i].name] = df_mock.loc[
            i, "idx ecFEF2575%ecFEV1"
        ]

    res_ve = var_elim.query(
        variables=[AR.name, HFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    hfev1_ve = res_ve[HFEV1.name].values
    ar_ve = res_ve[AR.name].values

    # Run custom model
    (
        fig,
        p_M_given_D,
        p_HFEV1_given_D,
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        light=False,
        n_days_consec=3,
    )
    hfev1_cc = p_HFEV1_given_D
    ar0_cc = AR_given_M_and_D[0, :]
    ar1_cc = AR_given_M_and_D[1, :]
    ar2_cc = AR_given_M_and_D[2, :]

    plot_diff(
        HFEV1,
        AR,
        hfev1_cc,
        hfev1_ve,
        ar0_cc,
        ar_ve,
        ar1_cc,
        ar_ve,
        ar2_cc,
        ar_ve,
    )

    assert_low_element_wise_max_diff(hfev1_cc, hfev1_ve)
    assert_low_element_wise_max_diff(ar0_cc, ar_ve)
    assert_low_element_wise_max_diff(ar1_cc, ar_ve)
    assert_low_element_wise_max_diff(ar2_cc, ar_ve)

    return None