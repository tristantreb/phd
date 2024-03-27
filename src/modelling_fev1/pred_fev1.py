import numpy as np
from scipy.stats import norm

import src.data.sanity_checks as sanity_checks


def calc_hfev1_prior(hfev1_bins, height, age, sex):
    """
    Computes the prior fo Healthy FEV1 given its bins
    This model uses the inversed LMS percentile curve function to compute the zscores of each bin given an input array of HFEV1/predictedFEV1 bin values
    """
    # Compute the predicted FEV1 for the individual
    pred_fev1_arr = calc_predicted_FEV1_LMS_straight(height, age, sex)
    S = pred_fev1_arr["S"]
    M = pred_fev1_arr["M"]
    L = pred_fev1_arr["L"]

    # Compute zscores for each bin
    zscores = get_inverse_lms_pred_fev1_for_zscore(hfev1_bins, S, M, L)

    # Get probabilities for each bin
    p = norm.pdf(zscores)
    return p / p.sum()


def calc_FEV1_prct_predicted_df(df):
    """
    Returns input DataFrame with FEV1 % Predicted as a new column, after sanity check
    """
    df["ecFEV1 % Predicted"] = df["ecFEV1"] / df["Predicted FEV1"] * 100
    df.apply(
        lambda x: sanity_checks.fev1_prct_predicted(x["ecFEV1 % Predicted"], x.ID),
        axis=1,
    )
    df["FEV1 % Predicted"] = df["FEV1"] / df["Predicted FEV1"] * 100
    df.apply(
        lambda x: sanity_checks.fev1_prct_predicted(x["FEV1 % Predicted"], x.ID), axis=1
    )
    return df


def calc_predicted_FEV1_linear(height: int, age: int, sex: str):
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


def calc_predicted_FEV1_LMS_df(df):
    """
    Returns a Series with Predicted FEV1 from a DataFrame with Sex, Height, Age
    """
    df["Predicted FEV1"] = df.apply(
        lambda x: calc_predicted_FEV1_LMS(
            load_LMS_spline_vals(x.Age, x.Sex),
            load_LMS_coeffs(x.Sex),
            x.Height,
            x.Age,
            x.Sex,
        )["M"],
        axis=1,
    )
    df.apply(lambda x: sanity_checks.predicted_fev1(x["Predicted FEV1"], x.ID), axis=1)
    return df


def calc_predicted_FEV1_LMS_straight(height: int, age: int, sex: str, debug=False):
    return calc_predicted_FEV1_LMS(
        load_LMS_spline_vals(age, sex),
        load_LMS_coeffs(sex),
        height,
        age,
        sex,
        debug,
    )


def get_lms_pred_fev1_for_zscore(zscore_arr, M, S, L, debug=False):
    x = np.exp(np.log(1 + zscore_arr * L * S) / L + np.log(M))
    if debug:
        print(
            f"LMS pred for zscore {zscore_arr} = exp(ln(1 + {zscore_arr} * {L} * {S}) / {L} + ln({M})) = {x}"
        )
    return x


def get_inverse_lms_pred_fev1_for_zscore(pred_fev1_arr, S, M, L):
    return (np.exp(L * np.log(pred_fev1_arr / M)) - 1) / (S * L)


def calc_predicted_FEV1_LMS(
    spline_vals, coeffs, height: int, age: int, sex: str, debug=False
):
    """
    Implemented from the GLI reference equations.
    The GLI's model a location (M), scale (S), shape (L) model to predicted FEV1 given a Z-Score​
    Equations: https://www.ers-education.org/lr/show-details/?idP=138978
    Paper: https://www.ersnet.org/science-and-research/ongoing-clinical-research-collaborations/the-global-lung-function-initiative/gli-tools/
    """
    # M = exp(a0 + a1*ln(Height) + a2*ln(Age) + a3*AfrAm + a4*NEAsia + a5*SEAsia + Mspline)
    M = np.exp(
        coeffs["M"]["Intercept"]
        + coeffs["M"]["Height"] * np.log(height)
        + coeffs["M"]["Age"] * np.log(age)
        + spline_vals["Mspline"]
    )

    # S =  exp(p0 + p1*ln(Age) + p2*AfrAm + p3*NEAsia + p4*SEAsia + Sspline)
    S = np.exp(
        coeffs["S"]["Intercept"]
        + coeffs["S"]["Age"] * np.log(age)
        + spline_vals["Sspline"]
    )

    L = coeffs["L"]["Intercept"] + coeffs["L"]["Age"] * np.log(age)
    if debug:
        print(
            f"M = exp({coeffs['M']['Intercept']} + {coeffs['M']['Height']}*ln({height}) + {coeffs['M']['Age']}*ln({age}) + {spline_vals['Mspline']})"
        )
        print(
            f"S = exp({coeffs['S']['Intercept']} + {coeffs['S']['Age']}*ln({age}) + {spline_vals['Sspline']})"
        )
        print(f"L = {coeffs['L']['Intercept']} + {coeffs['L']['Age']}*ln({age})")

    # Get lower limit of normal (5th percentile)
    LLN = get_lms_pred_fev1_for_zscore(-1.644, M, S, L, debug)

    # The Z-score of a value indicates how far from the mean is that value, in units of standard deviation.
    # In the LMS model, percentile_value(zscore) = exp(ln(1 - z-score *L*S)/L +ln(M))
    # Hence, the standard deviation is: percentile_value(zscore)
    n_std = 1
    sigma_up = get_lms_pred_fev1_for_zscore(n_std, M, S, L) - M
    sigma_down = M - get_lms_pred_fev1_for_zscore(-n_std, M, S, L)

    return {
        "M": M,
        "sigma_up": sigma_up,
        "sigma_down": sigma_down,
        "LLN": LLN,
        "L": L,
        "S": S,
    }


def load_LMS_spline_vals(age: int, sex: str):
    """
    Get the spline values for the M and S curves from the lookup table
    We ignore the Lspline column as it is always 0 for males or females
    """
    # Initially done reading from the lookup table directly.
    # 10x performance improve after hardcoding the values
    # (Breathe complementary study on O2 saturation took >14' to run on 200+ patients, then 1.2')

    # PATH_LMS_EQ_LOOKUP_TABLES = (
    #     "../../../../DataFiles/PredictedFEV1/FromGLI/GLI_LMS_equations_lookuptables.xls"
    # )
    # df = pd.read_excel(
    #     PATH_LMS_EQ_LOOKUP_TABLES,
    #     sheet_name=sheet_name,
    #     header=1,
    #     usecols="B:E",
    # )
    # Mspline = df[df.age == age].Mspline.values[0]
    # Sspline = df[df.age == age].Sspline.values[0]

    if sex == "Male":
        return _get_male_spline_vals(age)
    elif sex == "Female":
        return _get_female_spline_vals(age)
    else:
        raise ValueError(f"Sex {sex} not in Male/Female")


def _get_male_spline_vals(age: int):
    """
    Hardcoded values for the spline values for the M and S curves from the lookup table
    """
    if age < 3 or age > 95:
        raise ValueError(f"Age {age} not in 3-95 range")
    switch_dict = {
        3: {"Mspline": -0.1133230622, "Sspline": 0.2143372309},
        4: {"Mspline": -0.0892601186, "Sspline": 0.1793911904},
        5: {"Mspline": -0.0752420061, "Sspline": 0.1534965724},
        6: {"Mspline": -0.0787419775, "Sspline": 0.1284532609},
        7: {"Mspline": -0.0822026483, "Sspline": 0.0911586852},
        8: {"Mspline": -0.0778301622, "Sspline": 0.0439394795},
        9: {"Mspline": -0.0691240309, "Sspline": -0.0091507279},
        10: {"Mspline": -0.0549191373, "Sspline": -0.0460755428},
        11: {"Mspline": -0.0395487194, "Sspline": -0.0525338378},
        12: {"Mspline": -0.0175605902, "Sspline": -0.0386839924},
        13: {"Mspline": 0.0169429240, "Sspline": -0.0199492682},
        14: {"Mspline": 0.0612203578, "Sspline": -0.0090993724},
        15: {"Mspline": 0.1068262350, "Sspline": -0.0107926554},
        16: {"Mspline": 0.1460049430, "Sspline": -0.0234328133},
        17: {"Mspline": 0.1743911389, "Sspline": -0.0421490981},
        18: {"Mspline": 0.1923745447, "Sspline": -0.0614384988},
        19: {"Mspline": 0.2028810421, "Sspline": -0.0780782097},
        20: {"Mspline": 0.2080650726, "Sspline": -0.0900996230},
        21: {"Mspline": 0.2091123164, "Sspline": -0.0965358748},
        22: {"Mspline": 0.2071456302, "Sspline": -0.0980140546},
        23: {"Mspline": 0.2029590925, "Sspline": -0.0955504568},
        24: {"Mspline": 0.1970419230, "Sspline": -0.0905287643},
        25: {"Mspline": 0.1899386840, "Sspline": -0.0844055947},
        26: {"Mspline": 0.1821324805, "Sspline": -0.0782941738},
        27: {"Mspline": 0.1739285226, "Sspline": -0.0729294892},
        28: {"Mspline": 0.1655809617, "Sspline": -0.0685957509},
        29: {"Mspline": 0.1574146902, "Sspline": -0.0651994039},
        30: {"Mspline": 0.1494648728, "Sspline": -0.0625079453},
        31: {"Mspline": 0.1416572501, "Sspline": -0.0602809630},
        32: {"Mspline": 0.1339903349, "Sspline": -0.0584720126},
        33: {"Mspline": 0.1264626457, "Sspline": -0.0572000261},
        34: {"Mspline": 0.1190355393, "Sspline": -0.0566270901},
        35: {"Mspline": 0.1115898681, "Sspline": -0.0568020907},
        36: {"Mspline": 0.1040063042, "Sspline": -0.0575068610},
        37: {"Mspline": 0.0962413352, "Sspline": -0.0583879237},
        38: {"Mspline": 0.0883261182, "Sspline": -0.0591302235},
        39: {"Mspline": 0.0802801547, "Sspline": -0.0595716092},
        40: {"Mspline": 0.0720983260, "Sspline": -0.0596499962},
        41: {"Mspline": 0.0637753175, "Sspline": -0.0593401765},
        42: {"Mspline": 0.0553638050, "Sspline": -0.0585290440},
        43: {"Mspline": 0.0469393466, "Sspline": -0.0570427389},
        44: {"Mspline": 0.0385548346, "Sspline": -0.0547806711},
        45: {"Mspline": 0.0302199933, "Sspline": -0.0517233389},
        46: {"Mspline": 0.0218960234, "Sspline": -0.0478226597},
        47: {"Mspline": 0.0135116850, "Sspline": -0.0430534586},
        48: {"Mspline": 0.0050165364, "Sspline": -0.0374493139},
        49: {"Mspline": -0.0036398075, "Sspline": -0.0310786949},
        50: {"Mspline": -0.0124994715, "Sspline": -0.0240232453},
        51: {"Mspline": -0.0215737385, "Sspline": -0.0163759299},
        52: {"Mspline": -0.0308700995, "Sspline": -0.0082227480},
        53: {"Mspline": -0.0404160578, "Sspline": 0.0003347705},
        54: {"Mspline": -0.0502315535, "Sspline": 0.0091955803},
        55: {"Mspline": -0.0603055286, "Sspline": 0.0183045081},
        56: {"Mspline": -0.0706263907, "Sspline": 0.0276294168},
        57: {"Mspline": -0.0811909021, "Sspline": 0.0371546764},
        58: {"Mspline": -0.0919926314, "Sspline": 0.0469054986},
        59: {"Mspline": -0.1030183237, "Sspline": 0.0568912172},
        60: {"Mspline": -0.1142527680, "Sspline": 0.0670753983},
        61: {"Mspline": -0.1256800286, "Sspline": 0.0774166917},
        62: {"Mspline": -0.1372741752, "Sspline": 0.0878876913},
        63: {"Mspline": -0.1489968033, "Sspline": 0.0984517568},
        64: {"Mspline": -0.1608086778, "Sspline": 0.1090547636},
        65: {"Mspline": -0.1726866335, "Sspline": 0.1196217151},
        66: {"Mspline": -0.1846225540, "Sspline": 0.1300739110},
        67: {"Mspline": -0.1966076681, "Sspline": 0.1403566984},
        68: {"Mspline": -0.2086366843, "Sspline": 0.1504309080},
        69: {"Mspline": -0.2207005883, "Sspline": 0.1602637985},
        70: {"Mspline": -0.2327827516, "Sspline": 0.1698425218},
        71: {"Mspline": -0.2448646973, "Sspline": 0.1791697780},
        72: {"Mspline": -0.2569314373, "Sspline": 0.1882483377},
        73: {"Mspline": -0.2689718713, "Sspline": 0.1970812511},
        74: {"Mspline": -0.2809709301, "Sspline": 0.2056886818},
        75: {"Mspline": -0.2929127600, "Sspline": 0.2140926589},
        76: {"Mspline": -0.3047812815, "Sspline": 0.2223135327},
        77: {"Mspline": -0.3165604545, "Sspline": 0.2303731561},
        78: {"Mspline": -0.3282351430, "Sspline": 0.2382929351},
        79: {"Mspline": -0.3397948312, "Sspline": 0.2460907136},
        80: {"Mspline": -0.3512337601, "Sspline": 0.2537800562},
        81: {"Mspline": -0.3625472469, "Sspline": 0.2613722228},
        82: {"Mspline": -0.3737321756, "Sspline": 0.2688732240},
        83: {"Mspline": -0.3847871602, "Sspline": 0.2762877362},
        84: {"Mspline": -0.3957118470, "Sspline": 0.2836203910},
        85: {"Mspline": -0.4065078027, "Sspline": 0.2908735328},
        86: {"Mspline": -0.4171766366, "Sspline": 0.2980485321},
        87: {"Mspline": -0.4277204305, "Sspline": 0.3051452160},
        88: {"Mspline": -0.4381423480, "Sspline": 0.3121627926},
        89: {"Mspline": -0.4484460612, "Sspline": 0.3191017231},
        90: {"Mspline": -0.4586344998, "Sspline": 0.3259633849},
        91: {"Mspline": -0.4687101600, "Sspline": 0.3327489200},
        92: {"Mspline": -0.4786753400, "Sspline": 0.3394594460},
        93: {"Mspline": -0.4885325090, "Sspline": 0.3460968871},
        94: {"Mspline": -0.4982840794, "Sspline": 0.3526630378},
        95: {"Mspline": -0.5079323727, "Sspline": 0.3591595580},
    }
    return switch_dict.get(age)


def _get_female_spline_vals(age: int):
    """
    Hardcoded values for the spline values for the M and S curves from the lookup table
    Age between 18 and 60
    """
    if age < 3 or age > 95:
        raise ValueError(f"Age {age} not in 3-95 range")
    switch_dict = {
        3: {"Mspline": -0.2311081682, "Sspline": 0.3351484375},
        4: {"Mspline": -0.1817484965, "Sspline": 0.2438674473},
        5: {"Mspline": -0.1548415627, "Sspline": 0.1748737925},
        6: {"Mspline": -0.1451972063, "Sspline": 0.1208491509},
        7: {"Mspline": -0.1310395126, "Sspline": 0.0779905559},
        8: {"Mspline": -0.1124754957, "Sspline": 0.0451922130},
        9: {"Mspline": -0.0896807261, "Sspline": 0.0205607576},
        10: {"Mspline": -0.0573476736, "Sspline": 0.0036549657},
        11: {"Mspline": -0.0172083499, "Sspline": -0.0076252564},
        12: {"Mspline": 0.0273533022, "Sspline": -0.0159514572},
        13: {"Mspline": 0.0709281316, "Sspline": -0.0233420376},
        14: {"Mspline": 0.1086288153, "Sspline": -0.0304665211},
        15: {"Mspline": 0.1379293088, "Sspline": -0.0373050881},
        16: {"Mspline": 0.1582679761, "Sspline": -0.0437171291},
        17: {"Mspline": 0.1710660747, "Sspline": -0.0495223508},
        18: {"Mspline": 0.1784904743, "Sspline": -0.0546076352},
        19: {"Mspline": 0.1823234058, "Sspline": -0.0589703126},
        20: {"Mspline": 0.1839058476, "Sspline": -0.0627262980},
        21: {"Mspline": 0.1841007689, "Sspline": -0.0660492812},
        22: {"Mspline": 0.1831792840, "Sspline": -0.0691473190},
        23: {"Mspline": 0.1812498159, "Sspline": -0.0721568873},
        24: {"Mspline": 0.1784736511, "Sspline": -0.0751090059},
        25: {"Mspline": 0.1752865392, "Sspline": -0.0780069152},
        26: {"Mspline": 0.1720620703, "Sspline": -0.0808335603},
        27: {"Mspline": 0.1689525005, "Sspline": -0.0835765968},
        28: {"Mspline": 0.1658335981, "Sspline": -0.0862067807},
        29: {"Mspline": 0.1625464309, "Sspline": -0.0886779965},
        30: {"Mspline": 0.1590294730, "Sspline": -0.0909232135},
        31: {"Mspline": 0.1552696418, "Sspline": -0.0928791789},
        32: {"Mspline": 0.1512117498, "Sspline": -0.0944867020},
        33: {"Mspline": 0.1467492541, "Sspline": -0.0956848529},
        34: {"Mspline": 0.1418494365, "Sspline": -0.0964242640},
        35: {"Mspline": 0.1365080264, "Sspline": -0.0966705957},
        36: {"Mspline": 0.1307501791, "Sspline": -0.0963784902},
        37: {"Mspline": 0.1245841063, "Sspline": -0.0955024523},
        38: {"Mspline": 0.1180493758, "Sspline": -0.0940161230},
        39: {"Mspline": 0.1111860410, "Sspline": -0.0919156437},
        40: {"Mspline": 0.1040208072, "Sspline": -0.0892118205},
        41: {"Mspline": 0.0965858154, "Sspline": -0.0859200378},
        42: {"Mspline": 0.0889139012, "Sspline": -0.0820585307},
        43: {"Mspline": 0.0810515449, "Sspline": -0.0776626122},
        44: {"Mspline": 0.0730606465, "Sspline": -0.0727698043},
        45: {"Mspline": 0.0649877857, "Sspline": -0.0674043378},
        46: {"Mspline": 0.0568485480, "Sspline": -0.0615911744},
        47: {"Mspline": 0.0486138634, "Sspline": -0.0553610094},
        48: {"Mspline": 0.0402700231, "Sspline": -0.0487524488},
        49: {"Mspline": 0.0318067002, "Sspline": -0.0418033437},
        50: {"Mspline": 0.0231966359, "Sspline": -0.0345452885},
        51: {"Mspline": 0.0144120469, "Sspline": -0.0270087010},
        52: {"Mspline": 0.0054360064, "Sspline": -0.0192191547},
        53: {"Mspline": -0.0037443783, "Sspline": -0.0112016906},
        54: {"Mspline": -0.0131361813, "Sspline": -0.0029829828},
        55: {"Mspline": -0.0227368051, "Sspline": 0.0054092551},
        56: {"Mspline": -0.0325505985, "Sspline": 0.0139477141},
        57: {"Mspline": -0.0425906144, "Sspline": 0.0226112537},
        58: {"Mspline": -0.0528597065, "Sspline": 0.0313812287},
        59: {"Mspline": -0.0633529599, "Sspline": 0.0402366952},
        60: {"Mspline": -0.0740671946, "Sspline": 0.0491573819},
        64: {"Mspline": -0.1190269354, "Sspline": 0.0851666461},
        65: {"Mspline": -0.1307912904, "Sspline": 0.0941905083},
        66: {"Mspline": -0.1427713130, "Sspline": 0.1032036031},
        67: {"Mspline": -0.1549642350, "Sspline": 0.1121954971},
        68: {"Mspline": -0.1673585666, "Sspline": 0.1211539969},
        69: {"Mspline": -0.1799291385, "Sspline": 0.1300673621},
        70: {"Mspline": -0.1926478279, "Sspline": 0.1389254321},
        71: {"Mspline": -0.2054907403, "Sspline": 0.1477187029},
        72: {"Mspline": -0.2184336089, "Sspline": 0.1564402767},
        73: {"Mspline": -0.2314522525, "Sspline": 0.1650837905},
        74: {"Mspline": -0.2445224884, "Sspline": 0.1736429730},
        75: {"Mspline": -0.2576196994, "Sspline": 0.1821132656},
        76: {"Mspline": -0.2707177335, "Sspline": 0.1904927984},
        77: {"Mspline": -0.2837903596, "Sspline": 0.1987813317},
        78: {"Mspline": -0.2968135002, "Sspline": 0.2069790846},
        79: {"Mspline": -0.3097671325, "Sspline": 0.2150858256},
        80: {"Mspline": -0.3226324468, "Sspline": 0.2231016423},
        81: {"Mspline": -0.3353933817, "Sspline": 0.2310272332},
        82: {"Mspline": -0.3480405206, "Sspline": 0.2388635619},
        83: {"Mspline": -0.3605665884, "Sspline": 0.2466117359},
        84: {"Mspline": -0.3729667532, "Sspline": 0.2542729621},
        85: {"Mspline": -0.3852384709, "Sspline": 0.2618483965},
        86: {"Mspline": -0.3973798518, "Sspline": 0.2693391545},
        87: {"Mspline": -0.4093903970, "Sspline": 0.2767464143},
        88: {"Mspline": -0.4212709093, "Sspline": 0.2840714700},
        89: {"Mspline": -0.4354175164, "Sspline": 0.2970058329},
        90: {"Mspline": -0.4480929012, "Sspline": 0.3053970183},
        91: {"Mspline": -0.4607682861, "Sspline": 0.3137882036},
        92: {"Mspline": -0.4734436710, "Sspline": 0.3221793890},
        93: {"Mspline": -0.4861190558, "Sspline": 0.3305705744},
        94: {"Mspline": -0.4987944407, "Sspline": 0.3389617597},
        95: {"Mspline": -0.5114698256, "Sspline": 0.3473529451},
    }
    return switch_dict.get(age)


def load_LMS_coeffs(sex: str):
    """
    Get the coefficients for the L, M and S curves from the lookup table
    """
    # Initially done reading from the lookup table directly.
    # 10x performance improve after hardcoding the values
    # (Breathe complementary study on O2 saturation took >14' to run on 200+ patients, then 1.2')

    # PATH_LMS_EQ_LOOKUP_TABLES = (
    #     "../../../../DataFiles/PredictedFEV1/FromGLI/GLI_LMS_equations_lookuptables.xls"
    # )
    # df = pd.read_excel(
    #     PATH_LMS_EQ_LOOKUP_TABLES,
    #     sheet_name=sheet_name,
    #     header=2,
    #     usecols="G:I, K, L, N, O",
    #     index_col=0,
    #     nrows=7,
    #     names=["Coeff", "M_coeff", "M_val", "S_coeff", "S_val", "L_coeff", "L_val"],
    # )

    if sex == "Male":
        return {
            "M": {
                "Intercept": -10.342,
                "Height": 2.2196,
                "Age": 0.0574,
            },
            "S": {"Intercept": -2.3268, "Age": 0.0798},
            "L": {"Intercept": 0.8866, "Age": 0.085},
        }
    elif sex == "Female":
        return {
            "M": {
                "Intercept": -9.6987,
                "Height": 2.1211,
                "Age": -0.027,
            },
            "S": {"Intercept": -2.3765, "Age": 0.0972},
            "L": {"Intercept": 1.154, "Age": 0},
        }
    else:
        raise ValueError(f"Sex {sex} not in Male/Female")
