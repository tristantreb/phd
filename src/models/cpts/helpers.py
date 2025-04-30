from typing import List

import numpy as np
from plotly.subplots import make_subplots

import data.helpers as dh
import inference.helpers as ih
import models.helpers as mh


def get_cpt(vars: List[mh.VariableNode | mh.DiscreteVariableNode], suffix=None):
    path_to_folder = dh.get_path_to_src() + "models/cpts/"
    var_spec = map(
        lambda var: f"{mh.name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}", vars
    )
    filename = "_".join(var_spec)

    if suffix is not None:
        filename = filename + suffix

    cpt = np.load(f"{path_to_folder}{filename}.npy")

    assert len(vars) == len(cpt.shape)
    return cpt


def save_cpt(
    vars: List[mh.VariableNode | mh.DiscreteVariableNode], cpt: np.ndarray, suffix=None
):
    """
    Put the child variable first in the list of variables
    """
    path_to_folder = dh.get_path_to_src() + "/models/cpts/"
    filename = "_".join(
        [f"{mh.name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}" for var in vars]
    )

    if suffix is not None:
        filename = filename + suffix

    assert len(vars) == len(cpt.shape)

    np.save(
        f"{path_to_folder}{filename}",
        cpt,
    )
    return


def plot_2d_cpt(
    cpt,
    cVar,
    pVar,
    height=2500,
    vspace=0.003,
    invert=False,
    p_range=[0, 0.6],
    y_label_two_lines=False,
):
    """
    CPT represents P(cVar | pVar)
    The probability distribution of the child variable conditionned on the parent variable
    """
    if invert:
        cpt = cpt.T
        vartmp = cVar
        cVar = pVar
        pVar = vartmp

    plot_bins = range(pVar.card)
    title = f"CPT - P({cVar.get_abbr()}|{pVar.get_abbr()})"

    fig = make_subplots(
        rows=len(plot_bins), cols=1, shared_xaxes=True, vertical_spacing=vspace
    )

    for i, idx in enumerate(plot_bins):
        p = cpt[:, idx]
        ih.plot_histogram(
            fig, cVar, p, cVar.a, cVar.b, i + 1, 1, colour="#0072b2", annot=False
        )
        fig.update_yaxes(
            title_text=(
                f"{pVar.get_abbr()}<br>{pVar.midbins[idx]:2g}"
                if y_label_two_lines
                else f"{pVar.get_abbr()}={pVar.midbins[idx]:2g}"
            ),
            row=i + 1,
            col=1,
            range=p_range,
        )

    fig.update_xaxes(title_text=cVar.name, row=len(plot_bins), col=1)

    fig.update_layout(
        title=title,
        width=500,
        height=height,
        showlegend=False,
        font=dict(size=8),
    )
    return fig, title
