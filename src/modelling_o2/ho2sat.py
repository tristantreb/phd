def calc_healthy_O2_sat(height: int, sex: str):
    """
    Healthy/predicted O2 Saturation = a + b*isMale + c*(Height-avg_height)
    """
    # Results from fit done in breate_healthy_O2_modelling.ipynb
    a = 98.08683135024685
    b = -0.8209519722244375
    c = -0.017936971739351878
    avg_height = 166.54545454545453
    std = 0.5478455319170447

    if sex == "Female":
        return {
            "mean": a + c * (height - avg_height),
            "sigma": std,
        }
    elif sex == "Male":
        return {
            "mean": a + b + c * (height - avg_height),
            "sigma": std,
        }
    else:
        raise ValueError("Sex '{sex}' not in 'Female' or 'Male'")
