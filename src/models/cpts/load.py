import os

import numpy as np

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


def get_cpt(vars: [mh.variableNode]):
    filenames = map(
        lambda var: f"{name_to_abbr(var.name)}_{var.a}_{var.b}_{var.bin_width}", vars
    )
    filename = "_".join(filenames)

    path = os.getcwd().split("src")[0] + "src/models/cpts/" + filename + ".txt"

    # Check if filename exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    cpt = np.loadtxt(path, delimiter=",")
    return cpt
