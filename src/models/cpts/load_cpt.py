from typing import List

import numpy as np

import src.data.helpers as dh
import src.models.helpers as mh
from src.models.helpers import name_to_abbr


def get_cpt(vars: List[mh.VariableNode]):
    path_to_folder = dh.get_path_to_src() + "models/cpts/"
    var_spec = map(
        lambda var: f"{name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}", vars
    )
    filename = "_".join(var_spec)

    cpt = np.load(f"{path_to_folder}{filename}.npy")

    assert len(vars) == len(cpt.shape)
    return cpt


def save_cpt(vars: List[mh.VariableNode], cpt: np.ndarray):
    path_to_folder = dh.get_path_to_src() + "/models/cpts/"
    filename = "_".join(
        [f"{name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}" for var in vars]
    )

    assert len(vars) == len(cpt.shape)

    np.save(
        f"{path_to_folder}{filename}",
        cpt,
    )
    return
