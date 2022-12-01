def calc_predicted_fev1(row):
    # Calculate predicted FEV1 based on the formula from Andres
    # The input should be a df row with FEV1, Age, Height, Sex
    if row.Sex == "Male":
        return row.Height * 0.01 * 4.3 - row.Age * 0.029 - 2.49
    elif row.Sex == "Female":
        return row.Height * 0.01 * 3.95 - row.Age * 0.025 - 2.6
