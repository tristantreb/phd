import numpy as np
from pgmpy.inference.ExactInference import VariableElimination
from plotly.subplots import make_subplots

import src.inf_cutset_conditioning.cutset_cond_algs_learn_ar_change_noo2sat as cca_ar_change_noo2sat
import src.inference.helpers as ih
import src.models.builders as mb
import src.tests.data_factory as data


def test_light_model_no_o2sat():
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
    ecfev1_noise_model_suffix = "_std0.7"

    # Load variable eliminiation
    (
        model,
        HFEV1,
        AR_vars,
        uecFEV1_vars,
        ecFEV1_vars,
        ecFEF2575prctecFEV1_vars,
    ) = mb.fev1_fef2575_n_days_model_noise_shared_healthy_vars_and_temporal_ar_light(
        n_days,
        height,
        age,
        sex,
        ar_prior,
        ar_change_cpt_suffix,
        ecfev1_noise_model_suffix,
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
        log_p_D_given_M,
        AR_given_M_and_D,
        AR_given_M_and_all_D,
        res_dict,
    ) = cca_ar_change_noo2sat.run_long_noise_model_through_time(
        df_mock,
        ar_prior,
        ar_change_cpt_suffix=ar_change_cpt_suffix,
        ecfev1_noise_model_suffix=ecfev1_noise_model_suffix,
        light=True,
        n_days_consec=3,
    )
    hfev1_cc = p_M_given_D.reshape(-1)
    ar0_cc = AR_given_M_and_D[0, :]
    ar1_cc = AR_given_M_and_D[1, :]
    ar2_cc = AR_given_M_and_D[2, :]

    # Assert results are equal
    def get_element_wise_max_diff(v1, v2):
        return np.max(np.abs(v1 - v2))

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

    assert get_element_wise_max_diff(hfev1_cc, hfev1_ve) < 1e-8
    assert get_element_wise_max_diff(ar0_cc, ar0_ve) < 1e-8
    assert get_element_wise_max_diff(ar1_cc, ar1_ve) < 1e-8
    assert get_element_wise_max_diff(ar2_cc, ar2_ve) < 1e-8

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
