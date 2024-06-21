import logging
import math
from datetime import datetime
from re import A

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import src.data.helpers as dh

# from shared_code.utilities import calcPredictedFEV1, getMostRecentDataFrameFromFile
# from shared_code.blob_tools import BlobTools


# Todo: Validation steps when building tables
class RedCap:

    ml_tables = {}

    # def __init__(self):
    # self.blob = BlobTools()

    def load_redcap_id_map(self):
        """
        Load in latest red cap id map. It contains mapping from patient id values to updated id values
        """
        filename = (
            dh.get_path_to_main()
            + "DataFiles/BR/REDCapData/IDMappingFiles/PatientIDMappingFile-20240510.xlsx"
        )
        self.redcapidmap = pd.read_excel(filename, dtype={"ID": str, "redcap_id": str})

    def load_redcap_dropdown_dictionary(self):
        """
        Load in data dictionary for redcap data. Mainly used to replace drop down values with meaningful feature name
        """

        filename = (
            dh.get_path_to_main()
            + "DataFiles/BR/REDCapData/DataDictionary/AnalysisOfRemoteMonitoringVirt_DataDictionary_2023-11-29.csv"
        )
        self.redcapdict = pd.read_csv(filename)
        self.redcapdict_dropdown = self.redcapdict[
            self.redcapdict["Choices, Calculations, OR Slider Labels"].notna()
        ]

    def load_redcap_instrument_mapping(self):
        """
        Load in instrument field mapping. This is used to map field names in redcap to field names in br for instruments
        """

        filename = (
            dh.get_path_to_main()
            + "DataFiles/BR/REDCapData/FieldMapping/REDCapFieldMappingFile_20210616.xlsx"
        )
        self.redcap_instrument_map = pd.read_excel(
            filename, sheet_name="Instrument_Table"
        )

    def load_redcap_field_mapping(self):
        """
        Load in field mapping. This is used to map other field names in redcap to field names in br
        """

        filename = (
            dh.get_path_to_main()
            + "DataFiles/BR/REDCapData/FieldMapping/REDCapFieldMappingFile_20210616.xlsx"
        )
        self.redcap_field_map = pd.read_excel(filename, sheet_name="Field_Column")

    def load_raw_redcap_data(self):
        """
        Load in raw redcap data
        """

        filename = (
            dh.get_path_to_main()
            + "DataFiles/BR/REDCapData/DataExportFiles/AnalysisOfRemoteMoni_DATA_2024-05-10_1353.csv"
        )
        self.redcap_data = pd.read_csv(filename, dtype={"study_id": str})

    def addDropdownValues(self, replacement_columns):
        # where redcap_repeat_instrument is blank replace with 'patient_info'
        self.redcap_data["redcap_repeat_instrument"] = np.where(
            self.redcap_data["redcap_repeat_instrument"].isna(),
            "patient_info",
            self.redcap_data["redcap_repeat_instrument"],
        )

        # initialise a dictionary for mapping dropdown fields
        dd_dict = {}
        # subset dropdowns to only dropdown fields
        self.redcapdict_dropdown = self.redcapdict[
            self.redcapdict["Choices, Calculations, OR Slider Labels"].notna()
        ]
        # loop through the dictionary df and add to the dictionary dd_dict
        for r, row in self.redcapdict_dropdown.iterrows():
            if row["Variable / Field Name"] in replacement_columns:

                tmp_dict = {}
                all_options = row["Choices, Calculations, OR Slider Labels"].split(
                    " | "
                )
                for option in all_options:
                    key, value = option.split(", ", 1)
                    tmp_dict[float(key)] = value
                dd_dict[row["Variable / Field Name"]] = tmp_dict

        for col in replacement_columns:
            self.redcap_data[col + "_refactor"] = self.redcap_data[col].map(
                dd_dict[col]
            )

    def addHospitalData(self):
        # adding hospital and study number to every column
        id_merged = self.redcap_data.merge(
            self.redcapidmap, how="inner", left_on="study_id", right_on="redcap_id"
        )
        # Print IDs that are in redcapidmap but not in redcap_data
        print('IDs in redcapidmap but not in redcap_data', self.redcapidmap[~self.redcapidmap.redcap_id.isin(self.redcap_data.study_id)]['redcap_id'].unique())
        # Print IDs that are in redcap_data but not in redcapidmap
        print('IDs in redcap_data but not in redcapidmap', self.redcap_data[~self.redcap_data.study_id.isin(self.redcapidmap.redcap_id)]['study_id'].unique())
        tmpids = id_merged[
            id_merged["redcap_repeat_instrument"] == "patient_info"
        ].filter(["study_id", "hospital", "study_number"])
        mlhospcol = "Hospital"  # self.redcap_field_map[self.redcap_field_map["redcap_fieldname"]=="hospital"].iloc[0]["matlab_column"]
        mlstdynbrcol = "StudyNumber"  # self.redcap_field_map[self.redcap_field_map["redcap_fieldname"]=="study_number"].iloc[0]["matlab_column"]
        tmpids = tmpids.rename(
            columns={"hospital": mlhospcol, "study_number": mlstdynbrcol}
        )
        self.redcap_data = id_merged.merge(
            tmpids, how="left", left_on="study_id", right_on="study_id"
        )

    def outputMLTable(self, df: pd.DataFrame, name: str):
        datestamp = str(datetime.today().strftime("%Y%m%d"))
        df.to_csv(
            f"{dh.get_path_to_main()}DataFiles/BR/REDCapData/ProcessedData/{name}_{datestamp}.csv",
            index=False,
        )

    def listMLTables(self):
        print(f"Current ML tables are {self.ml_tables.keys()}")

    def produceTables(self):
        for i in self.redcap_instrument_map.iloc:
            # intialise instrument and table names
            rcinstr = i["redcap_instrument"]
            br_table_name = i["matlab_table"]

            # retrieve relevant records
            trcdata = self.redcap_data[
                self.redcap_data["redcap_repeat_instrument"] == rcinstr
            ].copy()

            temptable = (
                trcdata.filter(["study_id", "ID"]).reset_index().drop(columns=["index"])
            )
            trcdata["ID"] = [
                (
                    temptable.at[i, "ID"]
                    if str(temptable.at[i, "ID"]) != "nan"
                    else temptable.at[i, "study_id"]
                )
                for i in range(temptable.shape[0])
            ]

            trcdata = trcdata[trcdata[rcinstr + "_complete"] == 2]

            tfieldmap = self.redcap_field_map[
                self.redcap_field_map["redcap_instrument"] == rcinstr
            ]

            mltable_fields = ["ID", "Hospital", "StudyNumber"]

            mltable_fields += list(tfieldmap.redcap_fieldname.unique())

            print(f"{rcinstr} = {mltable_fields}")
            mltable = trcdata[mltable_fields].copy()

            rename_dict = (
                tfieldmap[["matlab_column", "redcap_fieldname"]]
                .set_index("redcap_fieldname")
                .to_dict()["matlab_column"]
            )

            mltable.rename(columns=rename_dict, inplace=True)
            self.ml_tables[br_table_name] = mltable
            # self.outputMLTable(mltable, br_table_name)

    def populateTableColumns(self):
        brPatient = self.ml_tables["brPatient"]

        for col in brPatient.columns:
            try:
                if brPatient[col].shape[1] == 2:
                    hold = brPatient[col].iloc[:, 0]
                    # logging.warning(f"{brPatient.columns}")
                    brPatient.drop(columns=[col], inplace=True)
                    # brPatient.Hospital = None
                    # logging.warning(f"{brPatient.columns}")
                    brPatient[col] = hold
            except:
                pass
            # logging.warning(f"{hold.shape}")
            # logging.warning(f"Hospital = {brPatient.Hospital.shape}")

        brPatient["StudyDate"] = pd.to_datetime(
            brPatient["StudyDate"], format="%Y-%m-%d"
        )
        brPatient["DOB"] = pd.to_datetime(brPatient["DOB"], format="%Y-%m-%d")
        brPatient["Prior6Mnth"] = [
            date - relativedelta(months=6) for date in brPatient["StudyDate"]
        ]
        brPatient["Post6Mnth"] = [
            date + relativedelta(months=6) for date in brPatient["StudyDate"]
        ]
        brPatient["PatClinDate"] = np.where(
            (brPatient["ConsentStatus"] == "Withdrawn")
            & (brPatient["WithdrawalDate"] < brPatient["PatClinDate"]),
            brPatient["WithdrawalDate"],
            brPatient["PatClinDate"],
        )
        # brPatient["FEV1SetAs"] = round(brPatient["PredictedFEV1"], 1)
        # logging.warning(f"Study numbers = {brPatient.StudyNumber.iloc[:,0].head()}")
        brPatient["StudyEmail"] = brPatient["StudyNumber"]  # .iloc[:,0]

        def get_delta_years(row):
            delta = relativedelta(row["StudyDate"], row["DOB"])
            return delta.years + delta.months / 12 + delta.days / 365.25

        brPatient["CalcAgeExact"] = brPatient.apply(get_delta_years, axis=1)
        brPatient["CalcAge"] = [
            math.floor(CalcAgeExact) for CalcAgeExact in brPatient["CalcAgeExact"]
        ]
        # brPatient["CalcPredictedFEV1"] = brPatient.apply(
        #     calcPredictedFEV1, axis=1, args=("CalcAge",)
        # )
        # brPatient["CalcPredictedFEV1OrigAge"] = brPatient.apply(
        #     calcPredictedFEV1, axis=1, args=("Age",)
        # )
        # brPatient["CalcFEV1SetAs"] = round(brPatient["CalcPredictedFEV1"], 1)
        # brPatient["CalcFEV1SetAsOrigAge"] = round(
        #     brPatient["CalcPredictedFEV1OrigAge"], 1
        # )
        self.ml_tables["brPatient"] = brPatient

        brPFT = self.ml_tables["brPFT"]
        brPFT["Units"] = "L"
        # logging.warning(f"Hospital = {brPatient.Hospital.shape}")

        brPFT = brPFT.merge(
            brPatient,
            how="left",
            left_on=["ID", "Hospital", "StudyNumber"],
            right_on=["ID", "Hospital", "StudyNumber"],
        )[
            [
                "ID",
                "Hospital",
                "StudyNumber",
                "LungFunctionDate",
                "FEV1",
                "Units",
                "Comments",
                # "FEV1SetAs",
                # "CalcFEV1SetAs",
            ]
        ]
        # brPFT["FEV1_"] = 100 * (brPFT["FEV1"] / brPFT["FEV1SetAs"])
        # brPFT["CalcFEV1_"] = 100 * (brPFT["FEV1"] / brPFT["CalcFEV1SetAs"])
        # brPFT["FEV1SetAs"] = np.nan
        # brPFT["CalcFEV1SetAs"] = np.nan
        self.ml_tables["brPFT"] = brPFT

        brCRP = self.ml_tables["brCRP"]
        brCRP["Units"] = "mg/L"
        brCRP["NumericLevel"] = brCRP["Level"]
        self.ml_tables["brCRP"] = brCRP

        brAntibiotics = self.ml_tables["brAntibiotics"]
        brAntibiotics["HomeIV_s"] = brAntibiotics["HomeIV_s"].fillna("No")
        self.ml_tables["brAntibiotics"] = brAntibiotics

    def runPreprocess(self):
        # load in all data
        self.load_redcap_id_map()
        self.load_raw_redcap_data()
        self.load_redcap_dropdown_dictionary()
        self.load_redcap_instrument_mapping()
        self.load_redcap_field_mapping()

        # columns to be replaced
        replacement_columns = [
            "mb_name",
            "up_type_of_contact",
            "ov_visit_type",
            "cv_location",
            "ab_name",
            "dt_name",
            "cfgene1",
            "cfgene2",
        ]

        self.addDropdownValues(replacement_columns)
        self.addHospitalData()
        self.produceTables()
        self.populateTableColumns()
        self.listMLTables()

    def outputAll(self):
        for name in self.ml_tables.keys():
            data = self.ml_tables[name]
            self.outputMLTable(data, name)


# tmp = RC.ml_tables['brPatient']
# tmp.apply(calcPredictedFEV1, args=(tmp["Age"],))


if __name__ == "__main__":
    RC = RedCap()
    RC.runPreprocess()
    RC.outputAll()


"""
Perform preprocessing mapping of fields and OUTPUT
"""
