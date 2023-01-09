def calc_predicted_fev1(height, age, sex):
    # Calculate predicted FEV1 based on the formula from Andres
    # This formula takes Age in years, Height in m and Sex
    # Input Height is in cm, hence the /100

    if sex == "Male":
        return (height / 100) * 4.3 - age * 0.029 - 2.49
    elif sex == "Female":
        return (height / 100) * 3.95 - age * 0.025 - 2.6
