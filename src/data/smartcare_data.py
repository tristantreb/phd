import pandas as pd

import data.helpers as dh


def load_o2_fev1_df_from_excel():
    path = dh.get_path_to_main() + "ExcelFiles/SC/SC_O2_FEV1.xlsx"
    df = pd.read_excel(path)
    # ID column as type string
    df["ID"] = df["ID"].astype(str)
    # Date Redocrded as datetime
    df["Date Recorded"] = df["Date Recorded"].dt.date
    return df
