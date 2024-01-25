import numpy as np
from plotly.subplots import make_subplots

import src.inference.helpers as ih
import src.modelling_o2.helpers as o2h
import src.models.helpers as mh
import src.models.point_in_time as model

sex = "Male"
age = 30
height = 175

def calc_cpts(sex: str, age: int, height: int):
    # TODO: why not int by default?
    height = int(height)
    age = int(age)
    hfev1_prior = {"type": "default", "height": height, "age": age, "sex": sex}
    ho2sat_prior = {
        "type": "default",
        "height": height,
        "sex": sex,
    }
    (HFEV1, FEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat) = model.calc_cpts(
        hfev1_prior, ho2sat_prior
    )

    # Encode variables
    HFEV1 = mh.encode_node_variable(HFEV1)
    FEV1 = mh.encode_node_variable(FEV1)
    AR = mh.encode_node_variable(AR)
    HO2Sat = mh.encode_node_variable(HO2Sat)
    O2SatFFA = mh.encode_node_variable(O2SatFFA)
    IA = mh.encode_node_variable(IA)
    UO2Sat = mh.encode_node_variable(UO2Sat)
    O2Sat = mh.encode_node_variable(O2Sat)

    return HFEV1, FEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat


def model_and_inference(
    HFEV1,
    ecFEV1,
    AR,
    HO2Sat,
    O2SatFFA,
    IA,
    UO2Sat,
    O2Sat,
    FEV1_obs: float,
    O2Sat_obs: float,
):
    """
    Decodes inputs from JSON format, build model, runs inference, and returns a figure
    """
    # Decode input vars
    HFEV1 = mh.decode_node_variable(HFEV1)
    ecFEV1 = mh.decode_node_variable(ecFEV1)
    AR = mh.decode_node_variable(AR)
    HO2Sat = mh.decode_node_variable(HO2Sat)
    O2SatFFA = mh.decode_node_variable(O2SatFFA)
    IA = mh.decode_node_variable(IA)
    UO2Sat = mh.decode_node_variable(UO2Sat)
    O2Sat = mh.decode_node_variable(O2Sat)

    # Build model
    _, inf_alg = model.build_pgmpy_model(
        HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat
    )

    # INFERENCE
    print("Inference user input: FEV1 =", FEV1_obs, ", O2Sat =", O2Sat_obs)

    q1 = ih.infer(
        inf_alg,
        [HFEV1, AR, HO2Sat, IA],
        [[ecFEV1, FEV1_obs], [O2Sat, O2Sat_obs]],
        show_progress=False,
        joint=False,
    )
    q2 = ih.infer(
        inf_alg,
        [AR, O2SatFFA, UO2Sat],
        [[ecFEV1, FEV1_obs], [O2Sat, O2Sat_obs]],
        show_progress=False,
        joint=False,
    )

    res_hfev1 = q1[HFEV1.name]
    res_ar = q1[AR.name]
    res_ho2sat = q1[HO2Sat.name]
    res_o2satffa = q2[O2SatFFA.name]
    res_ia = q1[IA.name]
    res_uo2sat = q2[UO2Sat.name]

    # PLOT
    # Priors take 1x1 cells, posteriors take 2x2 cells
    prior = {"type": "bar", "colspan": 2}
    posterior = {"type": "bar", "rowspan": 2, "colspan": 2}

    viz_layout = [
        [prior, None, None, None, prior, None],  # 1
        [posterior, None, None, None, posterior, None],  # 2
        [None, None, None, None, None, None],  # 3
        [None, None, prior, None, None, None],  # 4
        [None, None, posterior, None, None, None],  # 5
        [None, None, None, None, None, None],  # 6
        [None, None, None, None, posterior, None],  # 7
        [None, None, None, None, None, None],  # 8
        [None, None, posterior, None, None, None],  # 9
        [None, None, None, None, None, None],  # 10
        [None, None, None, None, posterior, None],  # 11
        [None, None, None, None, None, None],  # 12
        [None, None, None, None, None, None],  # 13
        [None, None, None, None, prior, None],  # 14
    ]

    fig = make_subplots(
        rows=np.shape(viz_layout)[0], cols=np.shape(viz_layout)[1], specs=viz_layout
    )

    fev1_min = ecFEV1.a
    fev1_max = ecFEV1.b
    o2sat_min = 80
    o2sat_max = 100
    ia_min = 0
    ia_max = 90

    # HFEV1
    ih.plot_histogram(fig, HFEV1, HFEV1.prior[:, 0], fev1_min, fev1_max, 1, 1, False)
    fig["data"][0]["marker"]["color"] = "green"

    ih.plot_histogram(fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1)
    fig["data"][1]["marker"]["color"] = "green"

    # HO2Sat
    ih.plot_histogram(
        fig, HO2Sat, HO2Sat.prior[:, 0], o2sat_min, o2sat_max, 1, 5, False
    )
    fig["data"][2]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.prior[:, 0]), 1, 5)

    ih.plot_histogram(fig, HO2Sat, res_ho2sat.values, o2sat_min, o2sat_max, 2, 5)
    fig["data"][3]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

    # AR
    ih.plot_histogram(fig, AR, AR.prior[:, 0], AR.a, AR.b, 4, 3, False)
    fig["data"][4]["marker"]["color"] = "crimson"

    ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3)
    fig["data"][5]["marker"]["color"] = "crimson"

    # O2SatFFA
    ih.plot_histogram(fig, O2SatFFA, res_o2satffa.values, o2sat_min, o2sat_max, 7, 5)
    fig["data"][6]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

    # IA
    ih.plot_histogram(fig, IA, res_ia.values, ia_min, ia_max, 9, 3)
    fig["data"][7]["marker"]["color"] = "crimson"

    # UO2Sat
    ih.plot_histogram(fig, UO2Sat, res_uo2sat.values, o2sat_min, o2sat_max, 11, 5)
    fig["data"][8]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, max(res_uo2sat.values), 11, 5)
    # Put the message up from O2Sat to UO2Sat to see the result from the generative o2sat noise model
    tmp_UO2Sat = UO2Sat
    tmp_UO2Sat.name = "Message up from O2Sat"
    # Given o2sat_obs, get the idx of the bin in which it falls in O2Sat
    o2sat_obs_idx = np.where(O2Sat.midbins == O2Sat_obs)[0][0]
    ih.plot_histogram(
        fig, tmp_UO2Sat, O2Sat.prior[o2sat_obs_idx, :], o2sat_min, o2sat_max, 14, 5
    )
    fig["data"][9]["marker"]["color"] = "blue"
    o2h.add_o2sat_normal_range_line(fig, O2Sat.prior[o2sat_obs_idx, :], 14, 5)

    fig.update_layout(
        showlegend=False, height=800, width=1400, font=dict(size=10), bargap=0.01
    )
    fig.update_traces(marker_line_width=0)

    return fig, ecFEV1.a, ecFEV1.b


def main():
    HFEV1, FEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat = calc_cpts(sex, age, height)
    fig, a, b = model_and_inference(
        HFEV1,
        FEV1,
        AR,
        HO2Sat,
        O2SatFFA,
        IA,
        UO2Sat,
        O2Sat,
        3.1,
        97,
    )
    return -1

main()

import cProfile
import pstats
import re

prof = cProfile.Profile()
prof.run("main()")
# prof.sort_stats('cumtime')
prof.dump_stats("output2.prof")

stream = open("output2.txt", "w")
stats = pstats.Stats("output2.prof", stream=stream)
stats.sort_stats("cumtime")
stats.print_stats()
