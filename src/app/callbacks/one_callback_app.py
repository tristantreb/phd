import numpy as np
from dash import Input, Output
from plotly.subplots import make_subplots

import src.app.assets.styles as s
import src.inference.helpers as ih
import src.modelling_o2.helpers as o2h
import src.modelling_o2.ia as ia
import src.models.builders as mb
import src.models.graph_builders as graph_builders
import src.models.var_builders as var_builders
from src.inference.inf_algs import apply_custom_bp


def build_fev1_o2sat_with_factor_graph(app):
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

        # INFERENCE
        print("Inference user input: FEV1 =", FEV1_obs, ", O2Sat =", O2Sat_obs)

        query = ih.infer_on_factor_graph(
            inf_alg,
            [HFEV1, AR, HO2Sat, IA, O2SatFFA, UO2Sat],
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
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, None, "green"
        )
        ih.plot_histogram(
            fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1, HFEV1.name, "green"
        )

        # HO2Sat
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, None, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.cpt), 1, 5)

        ih.plot_histogram(
            fig,
            HO2Sat,
            res_ho2sat.values,
            o2sat_min,
            o2sat_max,
            2,
            5,
            HO2Sat.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

        # AR
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, None, "crimson")

        ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3, AR.name, "crimson")

        # O2SatFFA
        ih.plot_histogram(
            fig,
            O2SatFFA,
            res_o2satffa.values,
            o2sat_min,
            o2sat_max,
            7,
            5,
            O2SatFFA.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

        # IA
        ih.plot_histogram(fig, IA, IA.cpt, ia_min, ia_max, 9, 3, None, "crimson")
        ih.plot_histogram(
            fig, IA, res_ia.values, ia_min, ia_max, 10, 3, IA.name, "crimson"
        )

        # UO2Sat
        ih.plot_histogram(
            fig,
            UO2Sat,
            res_uo2sat.values,
            o2sat_min,
            o2sat_max,
            12,
            5,
            UO2Sat.name,
            "blue",
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


def build_fev1_fef2575_o2sat_with_factor_graph(app):
    @app.callback(
        Output("lung-graph", "figure"),
        Output("FEF25-75-prct-FEV1-output", "children"),
        # Output("FEF25-75-slider", "min"),
        # Output("FEF25-75-slider", "max"),
        # Inputs
        Input("sex-select", "value"),
        Input("age-input", "value"),
        Input("height-input", "value"),
        # Evidences
        Input("FEV1-slider", "value"),
        Input("FEF25-75-slider", "value"),
        Input("O2Sat-slider", "value"),
    )
    def content(
        sex,
        age,
        height,
        FEV1_obs: float,
        FEF2575_obs: float,
        O2Sat_obs: float,
    ):
        (
            _,
            inf_alg,
            HFEV1,
            ecFEV1,
            AR,
            HO2Sat,
            O2SatFFA,
            IA,
            UO2Sat,
            O2Sat,
            ecFEF2575prctFEV1,
        ) = mb.o2sat_fev1_fef2575_point_in_time_model_shared_healthy_vars(
            height, age, sex
        )

        # INFERENCE
        print(
            "Inference user input: FEV1 =",
            FEV1_obs,
            ", FEF25-75 =",
            FEF2575_obs,
            ", O2Sat =",
            O2Sat_obs,
        )

        FEF2575prctFEV1_obs = FEF2575_obs / FEV1_obs * 100

        query = ih.infer_on_factor_graph(
            inf_alg,
            [HFEV1, AR, HO2Sat, IA, O2SatFFA, UO2Sat],
            [
                [ecFEV1, FEV1_obs],
                [ecFEF2575prctFEV1, FEF2575prctFEV1_obs],
                [O2Sat, O2Sat_obs],
            ],
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
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, None, "green"
        )
        ih.plot_histogram(
            fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1, HFEV1.name, "green"
        )

        # HO2Sat
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, None, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.cpt), 1, 5)

        ih.plot_histogram(
            fig,
            HO2Sat,
            res_ho2sat.values,
            o2sat_min,
            o2sat_max,
            2,
            5,
            HO2Sat.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

        # AR
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, None, "crimson")

        ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3, AR.name, "crimson")

        # O2SatFFA
        ih.plot_histogram(
            fig,
            O2SatFFA,
            res_o2satffa.values,
            o2sat_min,
            o2sat_max,
            7,
            5,
            O2SatFFA.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

        # IA
        ih.plot_histogram(fig, IA, IA.cpt, ia_min, ia_max, 9, 3, None, "crimson")
        ih.plot_histogram(
            fig, IA, res_ia.values, ia_min, ia_max, 10, 3, IA.name, "crimson"
        )

        # UO2Sat
        ih.plot_histogram(
            fig,
            UO2Sat,
            res_uo2sat.values,
            o2sat_min,
            o2sat_max,
            12,
            5,
            UO2Sat.name,
            "blue",
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

        return fig, f"FEF25-75 in % of FEV1: {FEF2575prctFEV1_obs:.2f}%"


def build_fev1_o2sat_with_bayes_net(app):
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
            mb.o2sat_fev1_point_in_time_model(height, age, sex)
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
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, None, "green"
        )
        ih.plot_histogram(
            fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1, HFEV1.name, "green"
        )

        # HO2Sat
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, None, "blue"
        )
        o2h.add_o2sat_normal_range_line(fig, max(HO2Sat.cpt), 1, 5)

        ih.plot_histogram(
            fig,
            HO2Sat,
            res_ho2sat.values,
            o2sat_min,
            o2sat_max,
            2,
            5,
            HO2Sat.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_ho2sat.values), 2, 5)

        # AR
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, None, "crimson")

        ih.plot_histogram(fig, AR, res_ar.values, AR.a, AR.b, 5, 3, AR.name, "crimson")

        # O2SatFFA
        ih.plot_histogram(
            fig,
            O2SatFFA,
            res_o2satffa.values,
            o2sat_min,
            o2sat_max,
            7,
            5,
            O2SatFFA.name,
            "blue",
        )
        o2h.add_o2sat_normal_range_line(fig, max(res_o2satffa.values), 7, 5)

        # IA
        ih.plot_histogram(fig, IA, IA.cpt, ia_min, ia_max, 9, 3, None, "crimson")
        ih.plot_histogram(
            fig, IA, res_ia.values, ia_min, ia_max, 10, 3, IA.name, "crimson"
        )

        # UO2Sat
        ih.plot_histogram(
            fig,
            UO2Sat,
            res_uo2sat.values,
            o2sat_min,
            o2sat_max,
            12,
            5,
            UO2Sat.name,
            "blue",
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


def build_fev1_o2sat_with_factor_graph_debug(app):
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
        # Var to infer
        Input("var-to-infer-select", "value"),
        Input("ia-prior-select", "value"),
    )
    def content(
        sex,
        age,
        height,
        FEV1_obs: float,
        O2Sat_obs: float,
        var_to_infer: str,
        ia_prior: str,
    ):
        (
            HFEV1,
            ecFEV1,
            AR,
            HO2Sat,
            O2SatFFA,
            IA,
            UO2Sat,
            O2Sat,
        ) = var_builders.o2sat_fev1_point_in_time_model_shared_healthy_vars(
            height, age, sex, ia_prior
        )

        model = graph_builders.fev1_o2sat_point_in_time_factor_graph(
            HFEV1, ecFEV1, AR, HO2Sat, O2SatFFA, IA, UO2Sat, O2Sat, check_model=False
        )
        inf_alg = apply_custom_bp(model)

        # Plot specs before inference
        fev1_min = ecFEV1.a
        fev1_max = ecFEV1.b
        o2sat_min = 80
        o2sat_max = 100
        ia_min = 0
        ia_max = 30

        # Update viz_layout with posterior plot location
        posterior_plot_location_dict = {
            HFEV1.name: (HFEV1, 2, 1, fev1_min, fev1_max),
            HO2Sat.name: (HO2Sat, 2, 5, o2sat_min, o2sat_max),
            AR.name: (AR, 5, 3, AR.a, AR.b),
            O2SatFFA.name: (O2SatFFA, 8, 5, o2sat_min, o2sat_max),
            IA.name: (IA, 11, 3, ia_min, ia_max),
            UO2Sat.name: (UO2Sat, 14, 5, o2sat_min, o2sat_max),
        }
        var_to_infer, post_row, post_col, xmin, xmax = posterior_plot_location_dict.get(
            var_to_infer
        )

        # INFERENCE
        print("Inference user input: FEV1 =", FEV1_obs, ", O2Sat =", O2Sat_obs)

        query, messages = ih.infer_on_factor_graph(
            inf_alg,
            [var_to_infer],
            [[ecFEV1, FEV1_obs], [O2Sat, O2Sat_obs]],
            get_messages=True,
        )

        # Plot
        # Priors take 1x1 cells, posteriors take 2x2 cells
        m_plot = {"type": "bar", "rowspan": 1, "colspan": 1}
        post_plot = {"type": "bar", "rowspan": 2, "colspan": 1}

        viz_layout = [
            [m_plot, None, None, None, m_plot],  # 1
            [None, None, None, None, None],  # 2
            [None, None, None, None, None],  #
            [m_plot, None, m_plot, None, m_plot],  # 3
            [None, m_plot, None, m_plot, None],  # 4
            [None, None, None, None, None],  # 5
            [m_plot, None, None, None, m_plot],  # 6
            [None, None, None, None, None],  # 7
            [None, None, None, None, None],  # 8
            [None, None, m_plot, None, m_plot],  # 9
            [None, None, None, m_plot, None],  # 10
            [None, None, None, None, None],  # 11
            [None, None, None, None, m_plot],  # 12
            [None, None, None, None, None],  # 13
            [None, None, None, None, None],  # 14
            [None, None, None, None, m_plot],  # 15
            [None, None, None, None, None],  # 16
            [None, None, None, None, None],  # 17
            [None, None, None, None, m_plot],  # 19
        ]

        # Update the layout
        viz_layout[post_row - 1][post_col - 1] = post_plot

        fig = make_subplots(
            rows=np.shape(viz_layout)[0], cols=np.shape(viz_layout)[1], specs=viz_layout
        )

        # Plot posterior
        ih.plot_histogram(
            fig,
            var_to_infer,
            query[var_to_infer.name].values,
            xmin,
            xmax,
            post_row,
            post_col,
            None,
            "black",
        )

        # Plot messages
        def get_key(factor_name, var_name):
            # The key is either "factor_name -> var_name" or "var_name -> factor_name"
            key = f"{factor_name} -> {var_name}"
            if key in messages:
                return key
            else:
                return f"{var_name} -> {factor_name}"

        # HFEV1 prior
        ih.plot_histogram(
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, None, "green"
        )
        # factor - HFEV1
        key = get_key(f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}']", HFEV1.name)
        ih.plot_histogram(
            fig, HFEV1, messages[key], fev1_min, fev1_max, 4, 1, None, "gray"
        )
        # FEV1 obs
        ih.plot_histogram(
            fig,
            ecFEV1,
            ecFEV1.get_point_message(FEV1_obs),
            fev1_min,
            fev1_max,
            7,
            1,
            None,
            "blue",
        )
        # AR - factor
        key = get_key(f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}']", AR.name)
        ih.plot_histogram(fig, AR, messages[key], AR.a, AR.b, 5, 2, None, "gray")
        # AR prior
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, AR.name, "crimson")
        # IA priors
        key = get_key(f"['{UO2Sat.name}', '{O2SatFFA.name}', '{IA.name}']", IA.name)
        ih.plot_histogram(
            fig, IA, IA.cpt.reshape(-1), ia_min, ia_max, 10, 3, IA.name, "crimson"
        )
        # factor - AR
        key = get_key(f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}']", AR.name)
        ih.plot_histogram(fig, AR, messages[key], AR.a, AR.b, 5, 4, None, "gray")
        # IA
        key = get_key(f"['{UO2Sat.name}', '{O2SatFFA.name}', '{IA.name}']", IA.name)
        ih.plot_histogram(fig, IA, messages[key], ia_min, ia_max, 11, 4, None, "gray")
        # HO2Sat prior
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, HO2Sat.name, "blue"
        )
        # HO2Sat to factor
        key = get_key(f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}']", HO2Sat.name)
        ih.plot_histogram(
            fig, HO2Sat, messages[key], o2sat_min, o2sat_max, 4, 5, None, "gray"
        )
        # O2SatFFA to factor
        key = get_key(
            f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}']", O2SatFFA.name
        )
        ih.plot_histogram(
            fig,
            O2SatFFA,
            messages[key],
            o2sat_min,
            o2sat_max,
            7,
            5,
            O2SatFFA.name,
            "gray",
        )
        # factor to node
        key = get_key(
            f"['{UO2Sat.name}', '{O2SatFFA.name}', '{IA.name}']", O2SatFFA.name
        )
        ih.plot_histogram(
            fig, O2SatFFA, messages[key], o2sat_min, o2sat_max, 10, 5, None, "gray"
        )
        # UO2Sat
        key = get_key(f"['{UO2Sat.name}', '{O2SatFFA.name}', '{IA.name}']", UO2Sat.name)
        ih.plot_histogram(
            fig, UO2Sat, messages[key], o2sat_min, o2sat_max, 13, 5, UO2Sat.name, "gray"
        )
        # factor to uO2Sat
        key = get_key(f"['{O2Sat.name}', '{UO2Sat.name}']", UO2Sat.name)
        ih.plot_histogram(
            fig, UO2Sat, messages[key], o2sat_min, o2sat_max, 16, 5, None, "gray"
        )
        # O2 sat
        key = get_key(f"['{O2Sat.name}', '{UO2Sat.name}']", O2Sat.name)
        ih.plot_histogram_discrete(
            fig, O2Sat, messages[key], o2sat_min, o2sat_max, 19, 5, None, "blue"
        )

        fig.update_layout(
            showlegend=False,
            height=1000,
            width=1400,
            font=dict(size=8),
            bargap=0.01,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig.update_traces(marker_line_width=0)

        return fig, ecFEV1.a, ecFEV1.b
