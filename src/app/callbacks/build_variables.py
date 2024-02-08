from dash import Input, Output

import src.models.helpers as mh
import src.models.var_builders as var_builders


def build_variables_callback(app):
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
    def build_variables(sex: str, age: int, height: int):
        (
            HFEV1,
            FEV1,
            AR,
            HO2Sat,
            O2SatFFA,
            IA,
            UO2Sat,
            O2Sat,
        ) = var_builders.o2sat_fev1_point_in_time(height, age, sex)

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
