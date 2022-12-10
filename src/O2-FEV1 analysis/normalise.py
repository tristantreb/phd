import numpy as np


def norm_by_stable_baseline(O2_FEV1, var_list):
    """
    Normalise O2_FEV1 vector by stable baseline
    :param O2_FEV1: dataframe with measurements
    :param var_list: columns on which to apply normalisation
    :return: updated O2_FEV1 dataframe with a new column with suffix "norm"
    """
    patients_ids = O2_FEV1.ID.unique()
    for var in var_list:
        O2_FEV1["{} norm".format(var)] = np.nan
        for id in patients_ids:
            mask_patient = O2_FEV1.ID == id
            O2_FEV1["{} norm".format(var)][mask_patient] = norm_individual_data(
                O2_FEV1[mask_patient], var, id
            )
    return O2_FEV1


# Compute patient stable baseline and normalise by it (mean)
def norm_individual_data(patient_O2_FEV1, var_to_normalise, id):
    # filter stable periods
    patient_O2_FEV1_stable = patient_O2_FEV1[patient_O2_FEV1["Is Exacerbated"] == False]
    if len(patient_O2_FEV1_stable) == 0:
        print("No stable measurements for id {}, filling with NaN".format(id))
        return np.nan
    else:
        # get avg and std of filtered values
        avg = patient_O2_FEV1_stable[var_to_normalise].mean()
        std = patient_O2_FEV1_stable[var_to_normalise].std()
        return patient_O2_FEV1[var_to_normalise].apply(lambda x: x - avg)
