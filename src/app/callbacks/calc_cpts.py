from dash import Dash, Input, Output, dcc, html

import src.models.helpers as mh
import src.models.point_in_time as model


def calc_cpts_callback(app):
    @app.callback(
        Output("HFEV1", "data"),
        Output("FEV1", "data"),
        Output("AR", "data"),
        Output("HO2Sat", "data"),
        Output("O2SatFFA", "data"),
        Output("IA", "data"),
        Output("UO2Sat", "data"),
        Output("O2Sat", "data"),
        # Inputs
        Input("sex-select", "value"),
        Input("age-input", "value"),
        Input("height-input", "value"),
    )
    def calc_cpts(sex: str, age: int, height: int):
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
