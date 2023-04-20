import bp
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


from lung_health_model import FEV1, C, U, inference

# Solving: "Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized" error
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H4("Interactive inference on the lung's health model"),
        dcc.Graph(id="graph"),
        html.P("FEV1:"),
        dcc.Slider(
            id="fev1",
            min=FEV1.bins[0],
            max=FEV1.bins[-2],
            value=3,
            marks={0: "0.2", (len(C.bins) - 1): "5.9"},
        ),
    ]
)

@app.callback(Output("graph", "figure"), Input("fev1", "value"))
def display(fev1):
    print("user input: FEV1 set to", fev1)

    [_fev1_bin, fev1_idx] = bp.get_bin_for_value(fev1, FEV1.bins)

    res_u = inference.query(variables=[U.name], evidence={FEV1.name: fev1_idx})
    res_c = inference.query(variables=[C.name], evidence={FEV1.name: fev1_idx})

    # create a figure with 2 subplots using plotly

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Bar(y=res_u.values, x=U.bins[:-1]), row=1, col=1)
    fig['layout']['xaxis']['title']=U.name

    fig.add_trace(go.Bar(y=res_c.values, x=C.bins[:-1]), row=1, col=2)
    fig['layout']['xaxis2']['title']=C.name
    # intbins=(C.bins*100).astype(int)
    # shortbins=np.delete(intbins, np.arange(1,intbins.size,2))
    # fig.update_layout(xaxis2=dict(tickvals=C.bins.tolist()))

    fig.update_layout(showlegend=False)
    return fig

app.run_server(debug=True, port=8049, use_reloader=False)
