import time

from pgmpy.inference import BeliefPropagation

import src.models.helpers as mh


def infer(
    inference_alg: BeliefPropagation,
    variables: tuple[mh.variableNode],
    evidences: tuple[tuple[mh.variableNode, float]],
    show_progress=True,
    joint=True,
):
    """
    Runs an inference query against a given PGMPY inference model, variables, evidences
    :param inference_alg: The inference algorithm to use
    :param variables: The variables to query
    :param evidences: The evidences to use

    :return: The result of the inference
    """
    var_names = [var.name for var in variables]

    evidences_binned = dict()
    for [evidence_var, value] in evidences:
        [_bin, bin_idx] = mh.get_bin_for_value(value, evidence_var.bins)
        evidences_binned.update({evidence_var.name: bin_idx})

    tic = time.time()
    query = inference_alg.query(
        variables=var_names,
        evidence=evidences_binned,
        show_progress=show_progress,
        joint=joint,
    )
    # print(f"Query took {time.time() - tic}s")

    return query
