import numpy as np
from dash import Input, Output
from plotly.subplots import make_subplots

import src.app.assets.styles as s
import src.inference.helpers as ih
import src.modelling_o2.helpers as o2h
import src.models.builders as mb
import src.modelling_o2.ia as ia


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
            [prior, None, None, None, prior],  # 1
            [posterior, None, None, None, posterior],  # 2
            [None, None, None, None, None, None, None, None],  # 3
            [None, None, None, None, None, prior, None, None],  # 4
            [None, None, None, None, None, posterior, None, None],  # 5
            [None, None, None, None, None, None, None, None],  # 6
            [None, None, None, None, None, None, None, posterior],  # 7
            [None, None, None, None, None, None, None, None],  # 8
            [None, None, None, None, None, prior, None, None],  # 8
            [None, None, None, None, None, posterior, None, None],  # 9
            [None, None, None, None, None, None, None, None],  # 10
            [None, None, None, None, None, None, None, posterior],  # 11
            [None, None, None, None, None, None, None, None],  # 12
            # [None,     None,     None,     None,     None,     None,     None,     None],  # 13
            # [None,     None,     None,     None,     None,     None,     None,     prior],  # 14
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


def build_all_with_bayes_net(app):
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
            [prior, None, None, None, prior],  # 1
            [posterior, None, None, None, posterior],  # 2
            [None, None, None, None, None, None, None, None],  # 3
            [None, None, None, None, None, prior, None, None],  # 4
            [None, None, None, None, None, posterior, None, None],  # 5
            [None, None, None, None, None, None, None, None],  # 6
            [None, None, None, None, None, None, None, posterior],  # 7
            [None, None, None, None, None, None, None, None],  # 8
            [None, None, None, None, None, prior, None, None],  # 8
            [None, None, None, None, None, posterior, None, None],  # 9
            [None, None, None, None, None, None, None, None],  # 10
            [None, None, None, None, None, None, None, posterior],  # 11
            [None, None, None, None, None, None, None, None],  # 12
            # [None,     None,     None,     None,     None,     None,     None,     None],  # 13
            # [None,     None,     None,     None,     None,     None,     None,     prior],  # 14
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


def build_all_with_factor_graph_debug(app):
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

        var_to_infer = HFEV1

        query, messages = ih.infer_on_factor_graph(
            inf_alg,
            [HFEV1],
            [[ecFEV1, FEV1_obs], [O2Sat, O2Sat_obs]],
            get_messages=True,
        )

        res_hfev1 = query[HFEV1.name]

        # PLOT
        # Priors take 1x1 cells, posteriors take 2x2 cells
        message = {"type": "bar", "rowspan": 1, "colspan": 1}
        posterior = {"type": "bar", "rowspan": 2, "colspan": 1}

        viz_layout = [
            [message, None, None, None, message],  # 1
            [posterior, None, None, None, None],  # 2
            [None, None, None, None, None],  #
            [message, None, message, None, message],  # 3
            [None, message, None, message, None],  # 4
            [None, None, None, None, None],  # 5
            [message, None, None, None, message],  # 6
            [None, None, None, None, None],  # 7
            [None, None, None, None, None],  # 8
            [None, None, message, None, message],  # 9
            [None, None, None, message, None],  # 10
            [None, None, None, None, None],  # 11
            [None, None, None, None, message],  # 12
            [None, None, None, None, None],  # 13
            [None, None, None, None, None],  # 14
            [None, None, None, None, message],  # 15
            [None, None, None, None, None],  # 16
            [None, None, None, None, None],  # 17
            [None, None, None, None, message],  # 19
        ]

        fig = make_subplots(
            rows=np.shape(viz_layout)[0], cols=np.shape(viz_layout)[1], specs=viz_layout
        )

        fev1_min = ecFEV1.a
        fev1_max = ecFEV1.b
        o2sat_min = 80
        o2sat_max = 100
        ia_min = 0
        ia_max = 30

        # HFEV1 prior
        ih.plot_histogram(
            fig, HFEV1, HFEV1.cpt, fev1_min, fev1_max, 1, 1, None, "green"
        )
        # hfev1 posterior
        ih.plot_histogram(
            fig, HFEV1, res_hfev1.values, fev1_min, fev1_max, 2, 1, None, "green"
        )
        # factor to node
        key = f"['{ecFEV1.name}', '{HFEV1.name}', '{AR.name}'] -> {HFEV1.name}"
        ih.plot_histogram(
            fig, HFEV1, messages[key], fev1_min, fev1_max, 4, 1, None, "green"
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
        key = "Airway resistance (%) -> ['ecFEV1 (L)', 'Healthy FEV1 (L)', 'Airway resistance (%)']"
        ih.plot_histogram(fig, AR, messages[key], AR.a, AR.b, 5, 2, None, "crimson")
        # AR prior
        ih.plot_histogram(fig, AR, AR.cpt, AR.a, AR.b, 4, 3, None, "crimson")
        # IA prior
        key = "Inactive alveoli (%) -> ['Underlying O2 saturation (%)', 'O2 saturation if fully functional alveoli (%)', 'Inactive alveoli (%)']"
        ih.plot_histogram(fig, IA, IA.cpt.reshape(-1), ia_min, ia_max, 10, 3, None, "crimson")
        # factor - AR
        key = f"['{O2SatFFA.name}', '{HO2Sat.name}', '{AR.name}'] -> {AR.name}"
        ih.plot_histogram(fig, AR, messages[key], AR.a, AR.b, 5, 4, None, "blue")
        # IA
        key = "Inactive alveoli (%) -> ['Underlying O2 saturation (%)', 'O2 saturation if fully functional alveoli (%)', 'Inactive alveoli (%)']"
        ih.plot_histogram(
            fig, IA, messages[key], ia_min, ia_max, 11, 4, None, "crimson"
        )
        # HO2Sat prior
        ih.plot_histogram(
            fig, HO2Sat, HO2Sat.cpt, o2sat_min, o2sat_max, 1, 5, None, "blue"
        )
        # HO2Sat to factor
        key = "Healthy O2 saturation (%) -> ['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)']"
        ih.plot_histogram(
            fig, HO2Sat, messages[key], o2sat_min, o2sat_max, 4, 5, None, "blue"
        )
        # O2SatFFA to factor
        key = f"{O2SatFFA.name} -> ['O2 saturation if fully functional alveoli (%)', 'Healthy O2 saturation (%)', 'Airway resistance (%)']"
        ih.plot_histogram(
            fig, O2SatFFA, messages[key], o2sat_min, o2sat_max, 7, 5, None, "blue"
        )
        # factor to node
        key = "['Underlying O2 saturation (%)', 'O2 saturation if fully functional alveoli (%)', 'Inactive alveoli (%)'] -> O2 saturation if fully functional alveoli (%)"
        ih.plot_histogram(
            fig, O2SatFFA, messages[key], o2sat_min, o2sat_max, 10, 5, None, "blue"
        )
        # UO2Sat
        key = "Underlying O2 saturation (%) -> ['Underlying O2 saturation (%)', 'O2 saturation if fully functional alveoli (%)', 'Inactive alveoli (%)']"
        ih.plot_histogram(
            fig, UO2Sat, messages[key], o2sat_min, o2sat_max, 13, 5, None, "blue"
        )
        # factor to uO2Sat
        key = "['O2 saturation (%)', 'Underlying O2 saturation (%)'] -> Underlying O2 saturation (%)"
        ih.plot_histogram(
            fig, UO2Sat, messages[key], o2sat_min, o2sat_max, 16, 5, None, "blue"
        )
        # O2 sat
        key = (
            "O2 saturation (%) -> ['O2 saturation (%)', 'Underlying O2 saturation (%)']"
        )
        ih.plot_histogram_discrete(
            fig, O2Sat, messages[key], o2sat_min, o2sat_max, 19, 5, None, "blue"
        )

        fig.update_layout(
            showlegend=False,
            height=1000,
            width=1400,
            font=dict(size=10),
            bargap=0.01,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig.update_traces(marker_line_width=0)

        return fig, ecFEV1.a, ecFEV1.b
