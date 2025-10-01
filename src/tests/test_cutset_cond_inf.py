import logging

import numpy as np
from pgmpy.inference.ExactInference import VariableElimination
from plotly.subplots import make_subplots

import inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat
import inference.helpers as ih
import models.builders as mb
import tests.data_factory as data


def assert_low_element_wise_max_diff(v1, v2, tol=1e-10):
    assert np.max(np.abs(v1 - v2)) < tol


def test_3d_fev1_fef2575_cutset_cond_inf_against_var_elim():
    """
    End to end test for 3 days model when observing 1/ FEV1, 2/ FEV1 and FEF2575.
    """
    for obs_mode in ["FEV1", "FEV1 and FEF2575"]:
        # Data
        df_mock = data.get_mock_data_3d()
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
            uFEV1_vars,
            ecFEV1_vars,
            ecFEF2575prctecFEV1_vars,
        ) = mb.fev1_fef2575_n_days_BN_noise_shared_healthy_vars_and_temporal_ar(
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

        evidence_dict = {}
        if obs_mode == "FEV1":
            df_mock = data.add_idx_obs_cols(df_mock, ecFEV1_vars[0])

            # Obs for var elim
            for i in range(n_days):
                evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]
        elif obs_mode == "FEV1 and FEF2575":
            df_mock = data.add_idx_obs_cols(
                df_mock, ecFEV1_vars[0], ecFEF2575prctecFEV1_vars[0]
            )

            # Obs for var elim
            for i in range(n_days):
                evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]
                evidence_dict[ecFEF2575prctecFEV1_vars[i].name] = df_mock.loc[
                    i, "idx ecFEF2575%ecFEV1"
                ]
        else:
            assert 0 == 1

        # Run variable elimination
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
            _,
            p_HFEV1_given_D,
            _,
            _,
            AR_given_M_and_D,
            _,
            _,
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

        # plot_diff_3_days(
        #     HFEV1,
        #     AR_vars[0],
        #     "Variable elimination",
        #     "Cutset conditioning",
        #     hfev1_ve,
        #     hfev1_cc,
        #     ar0_ve,
        #     ar0_cc,
        #     ar1_ve,
        #     ar1_cc,
        #     ar2_cc,
        #     ar2_ve,
        # )

        assert_low_element_wise_max_diff(hfev1_cc, hfev1_ve)
        assert_low_element_wise_max_diff(ar0_cc, ar0_ve)
        assert_low_element_wise_max_diff(ar1_cc, ar1_ve)
        assert_low_element_wise_max_diff(ar2_cc, ar2_ve)

    return None


def test_1d_fev1_fef2575_cutset_cond_inf_against_var_elim():
    """
    End to end test for 1 day model when observing 1/ FEV1, 2/ FEV1 and FEF2575.
    """
    for obs_mode in ["FEV1", "FEV1 and FEF2575"]:
        # for obs_mode in ["FEV1 and FEF2575"]:
        df_mock = data.get_mock_data_1d()
        height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]

        # Model parameters
        ar_prior = "uniform"
        ar_change_cpt_suffix = "_shape_factor_single_laplace_1.6"
        ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
        fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

        # Load variable eliminiation
        (
            model,
            HFEV1,
            AR,
            _,
            ecFEV1,
            ecFEF2575prctecFEV1,
        ) = mb.fev1_fef2575_1_day_BN_noise(
            height,
            age,
            sex,
            ar_prior=ar_prior,
            ar_fef2575_cpt_suffix=fef2575_cpt_suffix,
            ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        )
        var_elim = VariableElimination(model)

        evidence_dict = {}
        if obs_mode == "FEV1":
            df_mock = data.add_idx_obs_cols(df_mock, ecFEV1)

            # Obs for var elim
            evidence_dict[ecFEV1.name] = df_mock.loc[0, "idx ecFEV1 (L)"]
        elif obs_mode == "FEV1 and FEF2575":
            df_mock = data.add_idx_obs_cols(df_mock, ecFEV1, ecFEF2575prctecFEV1)

            # Obs for var elim
            evidence_dict[ecFEV1.name] = df_mock.loc[0, "idx ecFEV1 (L)"]
            evidence_dict[ecFEF2575prctecFEV1.name] = df_mock.loc[
                0, "idx ecFEF2575%ecFEV1"
            ]
        else:
            assert 0 == 1

        # Run variable elimination
        res_ve = var_elim.query(
            variables=[AR.name, HFEV1.name],
            evidence=evidence_dict,
            joint=False,
        )
        hfev1_ve = res_ve[HFEV1.name].values
        ar_ve = res_ve[AR.name].values

        # Run custom model
        (
            _,
            p_HFEV1_given_D,
            _,
            _,
            AR_given_M_and_D,
            _,
            _,
        ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
            df_mock,
            ar_prior=ar_prior,
            ar_change_cpt_suffix=ar_change_cpt_suffix,
            ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
            fef2575_cpt_suffix=fef2575_cpt_suffix,
        )
        hfev1_cc = p_HFEV1_given_D
        ar_cc = AR_given_M_and_D[0, :]

        assert_low_element_wise_max_diff(hfev1_cc, hfev1_ve)
        assert_low_element_wise_max_diff(ar_cc, ar_ve)

    return None


def plot_diff_3_days(
    HFEV1,
    AR,
    series1_name,
    series2_name,
    hfev1_ve,
    hfev1_cc,
    ar0_ve,
    ar0_cc,
    ar1_ve,
    ar1_cc,
    ar2_ve,
    ar2_cc,
    title_suffix="",
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
    title = f"{series1_name} (blue) vs {series2_name} (red)<br>{title_suffix}"
    fig.update_layout(showlegend=False, height=550, width=800, title=title)
    fig.show()


def plot_diff_n_days(
    HFEV1,
    AR,
    series1_name,
    series2_name,
    hfev1_1,
    hfev1_2,
    ars_1,
    ars_2,
    title_suffix="",
):
    fig = make_subplots(rows=len(ars_1) + 1, cols=1, vertical_spacing=0.13)
    # Add series 1
    ih.plot_histogram(fig, HFEV1, hfev1_1, HFEV1.a, HFEV1.b, 1, 1, annot=False)
    for i, ar_1 in enumerate(ars_1):
        ih.plot_histogram(fig, AR, ar_1, AR.a, AR.b, i + 2, 1, annot=False)

    # Add series 2
    ih.plot_histogram(fig, HFEV1, hfev1_2, HFEV1.a, HFEV1.b, 1, 1, title=HFEV1.name)
    for i, ar_2 in enumerate(ars_2):
        ih.plot_histogram(fig, AR, ar_2, AR.a, AR.b, i + 2, 1, annot=False)

    for i in range(len(ars_1) + 1):
        fig.data[i].marker.color = "#636EFA"
        fig.data[i + len(ars_1) + 1].marker.color = "#EF553B"
    # Reduce x axis title font size
    fig.update_xaxes(title_font=dict(size=12), title_standoff=7)

    # Hide legend
    title = f"{series1_name} (blue) vs {series2_name} (red)<br>{title_suffix}"
    fig.update_layout(
        showlegend=False, height=550, width=800, title=title, margin=dict(t=150)
    )
    fig.show()


def test_cutset_cond_gives_same_p_S_given_D_as_var_elim_with_fev1_in_evidence():
    df_mock = data.get_mock_data_3d()
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_single_laplace_card3"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    (
        model,
        _,
        _,
        _,
        ecFEV1_vars,
        _,
        S,
    ) = mb.fev1_fef2575_n_days_BN_noise_shared_healthy_vars_and_temporal_ar_learn_S(
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

    df_mock = data.add_idx_obs_cols(df_mock, ecFEV1_vars[0])
    evidence_dict = {}
    for i in range(n_days):
        evidence_dict[ecFEV1_vars[i].name] = df_mock.loc[i, "idx ecFEV1 (L)"]

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
        get_p_d_given_s=True,
    )

    p_S_given_D = np.exp(log_p_S_given_D - np.max(log_p_S_given_D))
    p_S_given_D /= np.sum(p_S_given_D)

    # Assert results are equal
    assert_low_element_wise_max_diff(p_S_given_D, s_ve)


def test_cutset_cond_gives_same_p_S_given_D_as_var_elim_with_fev1_and_fef2575_in_evidence():
    df_mock = data.get_mock_data_3d("changing")
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_single_laplace_card5"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"

    (
        model,
        _,
        _,
        _,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
        S,
    ) = mb.fev1_fef2575_n_days_BN_noise_shared_healthy_vars_and_temporal_ar_learn_S(
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
        get_p_d_given_s=True,
    )

    p_S_given_D = np.exp(log_p_S_given_D - np.max(log_p_S_given_D))
    p_S_given_D /= np.sum(p_S_given_D)

    # Assert results are equal
    assert_low_element_wise_max_diff(p_S_given_D, s_ve)


def test_1d_cutset_cond_gives_correct_p_fev1_given_fef2575():
    df_mock = data.get_mock_data_1d()
    height, age, sex = df_mock.iloc[0][["Height", "Age", "Sex"]]
    n_days = df_mock.shape[0]

    # Model parameters
    ar_prior = "uniform"
    ar_change_cpt_suffix = "_shape_factor_single_laplace_1.6"
    ecfev1_noise_model_suffix = "_std_add_mult_ecfev1"
    fef2575_cpt_suffix = "_ecfev1_2_days_model_add_mult_noise"

    # Load variable eliminiation
    (
        model,
        HFEV1,
        AR,
        _,
        ecFEV1,
        ecFEF2575prctecFEV1,
    ) = mb.fev1_fef2575_1_day_BN_noise(
        height,
        age,
        sex,
        ar_prior=ar_prior,
        ar_fef2575_cpt_suffix=fef2575_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
    )
    var_elim = VariableElimination(model)

    df_mock = data.add_idx_obs_cols(df_mock, ecFEV1, ecFEF2575prctecFEV1)
    evidence_dict = {}
    evidence_dict[ecFEF2575prctecFEV1.name] = df_mock.loc[0, "idx ecFEF2575%ecFEV1"]

    # P(ecFEV1 | M, ecFEF2575)
    res_ve = var_elim.query(
        variables=[ecFEV1.name],
        evidence=evidence_dict,
        joint=False,
    )
    dist_ecfev1_ve = res_ve[ecFEV1.name].values
    p_ecfev1_ve = dist_ecfev1_ve[df_mock.loc[0, "idx ecFEV1 (L)"]]

    # Run custom model
    ([log_p_FEV1_given_S], _) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior=ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        fef2575_cpt_suffix=fef2575_cpt_suffix,
        get_p_fev1_given_s=True,
    )
    p_ecfev1_cc = np.exp(log_p_FEV1_given_S)

    assert_low_element_wise_max_diff(p_ecfev1_ve, p_ecfev1_cc)
