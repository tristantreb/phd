import numpy as np


# Normalise by patients' baseline lung function, which is the avg of stable measurements for that patient
def normalise_by_stable_baseline(O2_FEV1, lung_function_var):
    patients_ids = O2_FEV1.ID.unique()
    O2_FEV1["Norm Lung Function"] = np.nan
    for id in patients_ids:
        mask_patient = O2_FEV1.ID == id
        O2_FEV1["Norm Lung Function"][mask_patient] = normalise_by_patient_stable_baseline(O2_FEV1[mask_patient],
                                                                                           lung_function_var, id)
    return O2_FEV1


def normalise_by_patient_stable_baseline(patient_O2_FEV1, lung_function_var, id):
    # filter stable lung function
    patient_O2_FEV1_stable = patient_O2_FEV1[patient_O2_FEV1["Is Exacerbated"] == False]
    print(id, len(patient_O2_FEV1_stable))
    # print(patient_O2_FEV1_stable[['ID', 'FEV1', 'Is Exacerbated', 'Norm Lung Function']])
    if len(patient_O2_FEV1_stable) == 0:
        print("No stable measurements for id {}, filling with NaN".format(id))
        return np.nan
    else:
        # get avg and std of filtered values
        avg = patient_O2_FEV1_stable[lung_function_var].mean()
        std = patient_O2_FEV1_stable[lung_function_var].std()
        print(avg, std)
        match std:
            # normalise by mean
            case np.nan:
                return patient_O2_FEV1[lung_function_var].apply(lambda x: x - avg)
            # normalise by mean and std dev
            case _:
                return patient_O2_FEV1[lung_function_var].apply(lambda x: (x - avg) / std)
