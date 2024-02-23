from typing import List

import numpy as np

import src.data.helpers as dh
import src.models.helpers as mh


def name_to_abbr(name: str):
    map = {
        "Healthy FEV1 (L)": "HFEV1",
        "ecFEV1 (L)": "ecFEV1",
        "Airway resistance (%)": "AR",
        "O2 saturation (%)": "O2Sat",
        "Healthy O2 saturation (%)": "HO2Sat",
        "O2 saturation if fully functional alveoli (%)": "O2SatFFA",
        "Inactive alveoli (%)": "IA",
        "Underlying O2 saturation (%)": "UO2Sat",
    }

    abbr = map.get(name, "Invalid name")
    if abbr == "Invalid name":
        raise ValueError(f"Invalid name: {name}")

    return abbr


def get_cpt_2D(vars: List[mh.variableNode]):
    """
    Function specific to PGMPY which takes in only 2D arrays.
    Obsolete: Use get_cpt instead and reshape your array upon loading if necessary.
    """
    var_spec = map(
        lambda var: f"{name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}", vars
    )
    filename = "_".join(var_spec)
    path = dh.get_path_to_src() + "models/cpts/" + filename + ".txt"

    cpt = np.loadtxt(path, delimiter=",")

    assert 2 == len(cpt.shape)
    return cpt


def get_cpt(vars: List[mh.variableNode]):
    path_to_folder = dh.get_path_to_src() + "models/cpts/"
    var_spec = map(
        lambda var: f"{name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}", vars
    )
    filename = "_".join(var_spec)

    cpt = np.load(f"{path_to_folder}{filename}.npy")

    assert len(vars) == len(cpt.shape)
    return cpt


def save_cpt(vars: List[mh.variableNode], cpt: np.ndarray):
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
