def calc_predicted_fev1(height: int, age: int, sex: str):
    """
    Calculate predicted FEV1 according to the formula given by the lung function people at Royal Papworth Hospital
    This formula takes Age in years, Height in m and Sex
    Input Height is in cm, hence the /100
    """

    if sex == "Male":
        pred_FEV1 = (height / 100) * 4.3 - age * 0.029 - 2.49
        std_dev = 0.4
        return {"Predicted FEV1": pred_FEV1, "std": std_dev}
    elif sex == "Female":
        pred_FEV1 = (height / 100) * 3.95 - age * 0.025 - 2.6
        std_dev = 0.35
        return {"Predicted FEV1": pred_FEV1, "std": std_dev}
