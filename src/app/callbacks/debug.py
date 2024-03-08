import numpy as np
from dash import Input, Output
from plotly.subplots import make_subplots

import src.app.assets.styles as s
import src.inference.helpers as ih
import src.modelling_o2.helpers as o2h
import src.models.builders as mb


def build_all_with_factor_graph(app):
    @app.callback(
        Output("lung-graph", "figure"),
        Output("FEV1-slider", "min"),
        Output("FEV1-slider", "max"),
        # Inputs
        Input("sex-select", "value"),
        Input("age-input", "value"),
        Input("height-input", "value"),
        # Evidences
        Input("FEV1-slider", "value"),
        Input("O2Sat-slider", "value"),
    )
    def content(
        sex,
        age,
        height,
        FEV1_obs: float,
        O2Sat_obs: float,
    ):

        _, inf_alg, HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat = (
            mb.o2sat_fev1_point_in_time_model_shared_healthy_vars(height, age, sex)
        )

        var_to_infer = HO2Sat.name

        # INFERENCE
        print("Inference user input: FEV1 =", FEV1_obs, ", O2Sat =", O2Sat_obs)

        query = ih.infer_on_factor_graph(
            inf_alg,
            [var_to_infer],
            [[ecFEV1, FEV1_obs], [O2Sat, O2Sat_obs]],
        )

        res_hfev1 = query[HFEV1.name]
        res_ar = query[AR.name]
        res_ho2sat = query[HO2Sat.name]
        res_o2satffa = query[O2SatFFA.name]
        res_ia = query[IA.name]
        res_uo2sat = query[UO2Sat.name]

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
            [None, None, prior, None, None, None],  # 8
            [None, None, posterior, None, None, None],  # 9
            [None, None, None, None, None, None],  # 10
            [None, None, None, None, posterior, None],  # 11
            [None, None, None, None, None, None],  # 12
            # [None, None, None, None, None, None],  # 13
            # [None, None, None, None, prior, None],  # 14
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
        ih.plot_histogram(
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, False, "green"
        )
        ih.plot_histogram(
            fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1, True, "green"
        )

        # HO2Sat
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, False, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.cpt), 1, 5)

        ih.plot_histogram(
            fig, HO2Sat, res_ho2sat.values, o2sat_min, o2sat_max, 2, 5, True, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

        # AR
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, False, "crimson")

        ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3, True, "crimson")

        # O2SatFFA
        ih.plot_histogram(
            fig, O2SatFFA, res_o2satffa.values, o2sat_min, o2sat_max, 7, 5, True, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

        # IA
        ih.plot_histogram(fig, IA, IA.cpt, ia_min, ia_max, 9, 3, False, "crimson")
        ih.plot_histogram(
            fig, IA, res_ia.values, ia_min, ia_max, 10, 3, True, "crimson"
        )

        # UO2Sat
        ih.plot_histogram(
            fig, UO2Sat, res_uo2sat.values, o2sat_min, o2sat_max, 12, 5, True, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_uo2sat.values), 12, 5)

        fig.update_layout(
            showlegend=False,
            height=600,
            width=1400,
            font=dict(size=10),
            bargap=0.01,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig.update_traces(marker_line_width=0)

        return fig, ecFEV1.a, ecFEV1.b
