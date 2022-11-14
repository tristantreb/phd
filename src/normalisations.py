import numpy as np


# Normalise by stable baseline for a given variable
def normalise_by_stable_baseline(O2_FEV1, var_to_normalise):
    patients_ids = O2_FEV1.ID.unique()
    O2_FEV1["{} norm".format(var_to_normalise)] = np.nan
    for id in patients_ids:
        mask_patient = O2_FEV1.ID == id
        O2_FEV1["{} norm".format(var_to_normalise)][mask_patient] = normalise_by_patient_stable_baseline(O2_FEV1[mask_patient],
                                                                                           var_to_normalise, id)
    return O2_FEV1


# Compute patient stable baseline and normalise by it (mean)
def normalise_by_patient_stable_baseline(patient_O2_FEV1, var_to_normalise, id):
    # filter stable periods
    patient_O2_FEV1_stable = patient_O2_FEV1[patient_O2_FEV1["Is Exacerbated"] == False]
    if len(patient_O2_FEV1_stable) == 0:
        print("No stable measurements for id {}, filling with NaN".format(id))
        return np.nan
    else:
        # get avg and std of filtered values
        avg = patient_O2_FEV1_stable[var_to_normalise].mean()
        std = patient_O2_FEV1_stable[var_to_normalise].std()
        match std:
            # normalise by mean
            case np.nan:
                return patient_O2_FEV1[var_to_normalise].apply(lambda x: x - avg)
            # normalise by mean and std dev
            case _:
                return patient_O2_FEV1[var_to_normalise].apply(lambda x: (x - avg))
                # return patient_O2_FEV1[var_to_normalise].apply(lambda x: (x - avg) / std)
