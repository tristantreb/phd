import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import src.data.breathe_data as br
import src.data.helpers as dh

# from shared_code.utilities import getMostRecentDataFrameFromFile, normalRound, toDateNum


class MeasurementData:

    def __init__(self):
        patient_loc = dh.get_path_to_main() + "DataFiles/BR/PredModInputData.xlsx"
        logging.warning(f"Loading patient data from {patient_loc}")
        self.brPatient = br.load_patient_df_from_excel()
        self.brphysdata = pd.DataFrame(
            columns=[
                "SmartCareID",
                "ScaledDateNum",
                "DateNum",
                "UserName",
                "RecordingType",
                "CaptureType",
                "Date_TimeRecorded",
                "FEV",
                "WeightInKg",
                "O2Saturation",
                "Pulse_BPM_",
                "Temp_degC_",
                "Calories",
                "Rating",
                "Sleep",
                "HasCondition",
                # "CalcFEV1_",
            ]
        )
        self.brphysdata_deleted = pd.DataFrame(
            columns=[
                "SmartCareID",
                "ScaledDateNum",
                "DateNum",
                "UserName",
                "RecordingType",
                "CaptureType",
                "Date_TimeRecorded",
                "FEV",
                "WeightInKg",
                "O2Saturation",
                "Pulse_BPM_",
                "Temp_degC_",
                "Calories",
                "Rating",
                "Sleep",
                "HasCondition",
                # "CalcFEV1_",
                "Reason",
            ]
        )
        self.inputcolnames = {
            "CoughRecording": "Value",
            "TemperatureRecording": "Value",
            "WeightRecording": "Value",
            "WellnessRecording": "Value",
            "CalorieRecording": "Calories",
            "FEV1Recording": "FEV1",
            "FEF2575Recording": "FEF2575",
            "PEFRecording": "PEF",
            "FEV075Recording": "FEV075",
            "FEV1DivFEV6Recording": "FEV1DivFEV6",
            "FEV6Recording": "FEV6",
            "HasColdOrFluRecording": "HasColdOrFlu",
            "HasHayFeverRecording": "HasHayFever",
            "MinsAsleepRecording": "TotalMinutesAsleep",
            "MinsAwakeRecording": "Wake",
            "O2SaturationRecording": "SpO2",
            "PulseRateRecording": "HeartRate",
            "RestingHRRecording": "RestingHeartRate",
        }
        self.outputcolnames = {
            "ActivityRecording": "Activity_Steps",
            "AppetiteRecording": "Rating",
            "BreathlessnessRecording": "Rating",
            "CoughRecording": "Rating",
            "SleepActivityRecording": "Rating",
            "SputumVolumeRecording": "Rating",
            "TirednessRecording": "Rating",
            "WellnessRecording": "Rating",
            "CalorieRecording": "Calories",
            "FEV1Recording": "FEV",
            "FEF2575Recording": "FEV",
            "PEFRecording": "FEV",
            "FEV075Recording": "FEV",
            "FEV1DivFEV6Recording": "FEV",
            "FEV6Recording": "FEV",
            "InterpFEV1Recording": "FEV",
            "HasColdOrFluRecording": "HasCondition",
            "HasHayFeverRecording": "HasCondition",
            # "LungFunctionRecording": "CalcFEV1_",
            "MinsAsleepRecording": "Sleep",
            "MinsAwakeRecording": "Sleep",
            "O2SaturationRecording": "O2Saturation",
            "PulseRateRecording": "Pulse_BPM_",
            "RestingHRRecording": "Pulse_BPM_",
            "RespiratoryRateRecording": "BreathsPerMin",
            "SleepDisturbanceRecording": "NumSleepDisturb",
            "SputumColorRecording": "SputumColour",
            "SputumColourRecording": "SputumColour",
            "SputumSampleRecording": "SputumSampleTaken_",
            "TemperatureRecording": "Temp_degC_",
            "WeightRecording": "WeightInKg",
            "InterpWeightRecording": "WeightInKg",
        }

    def loadMeasureTables(self):
        """
        Function to load in measurement data from storage and join with the brPatient REDCap data.
        """
        self.measure_tables = {}
        for meas_table_name in [
            # "Activity",
            # "Coughing",
            # "HeartRate",
            "Oximeter",
            # "Sleep",
            "Spirometer",
            # "Temperature",
            # "Weight",
            # "Wellbeing",
        ]:
            file_location = (
                dh.get_path_to_main()
                + f"DataFiles/BR/MeasurementData/Breathe_{meas_table_name}_20231113.csv"
            )
            self.measure_tables[meas_table_name] = pd.read_csv(file_location)
            self.measure_tables[meas_table_name] = self.measure_tables[
                meas_table_name
            ].merge(
                self.brPatient.filter(
                    ["ID", "StudyNumber", "StudyDate", "PatClinDate", "PartitionKey"]
                ),
                how="left",
                left_on="UserId",
                right_on="PartitionKey",
            )
            aware_timestamp = pd.to_datetime(
                self.measure_tables[meas_table_name]["ClientTimestamp"], utc=False
            )
            unaware_timestamp = aware_timestamp.apply(lambda x: x.astimezone(None))
            self.measure_tables[meas_table_name]["TimestampDt"] = aware_timestamp
            self.measure_tables[meas_table_name]["DateDt"] = unaware_timestamp
            self.measure_tables[meas_table_name] = self.measure_tables[meas_table_name][
                (self.measure_tables[meas_table_name]["StudyNumber"].notnull())
                & (self.measure_tables[meas_table_name]["ID"].notnull())
                & (self.measure_tables[meas_table_name]["IsDeleted"] == False)
                # TODO: updated on 29.11.2023
                # (self.measure_tables[meas_table_name]["DateDt"] >= self.measure_tables[meas_table_name]["StudyDate"]) &
                # (self.measure_tables[meas_table_name]["DateDt"] <= self.measure_tables[meas_table_name]["PatClinDate"])
            ]

    def addBreatheRowsForMeasure(self, meas_table, recordingtype, delzero):
        """Function to preprocess the different measurement tables and add in the breathe rows.

        if delzero, then delete rows where the value is zero, otherwise just delete null rows

        Args:
            meas_table (DataFrame): measurement data under consideration
            recordingtype (str): recording type
            delzero (bool): defines how to handle nulls and zero values
        """
        temp_physdata = pd.DataFrame(
            columns=[
                "SmartCareID",
                "ScaledDateNum",
                "DateNum",
                "UserName",
                "RecordingType",
                "CaptureType",
                "Date_TimeRecorded",
                "FEV",
                "WeightInKg",
                "O2Saturation",
                "Pulse_BPM_",
                "Temp_degC_",
                "Calories",
                "Rating",
                "Sleep",
                "HasCondition",
                # "CalcFEV1_",
            ]
        )
        if meas_table.shape[0] > 0:
            temp_physdata["SmartCareID"] = meas_table["ID"]
            temp_physdata["UserName"] = meas_table["StudyNumber"]
            temp_physdata["Date_TimeRecorded"] = meas_table["DateDt"]
            temp_physdata["CaptureType"] = meas_table["CaptureType"]
            temp_physdata["RecordingType"] = recordingtype
            temp_physdata[self.outputcolnames[recordingtype]] = meas_table[
                self.inputcolnames[recordingtype]
            ]
            if recordingtype == "WellnessRecording":
                temp_physdata[self.outputcolnames[recordingtype]] = (
                    temp_physdata[self.outputcolnames[recordingtype]] * 10
                )
            elif recordingtype == "CoughRecording":
                temp_physdata[self.outputcolnames[recordingtype]] = 100 - (
                    temp_physdata[self.outputcolnames[recordingtype]] * 10
                )
            if isinstance(
                (temp_physdata[self.outputcolnames[recordingtype]].iloc[0]),
                (int, float, np.float64, np.int64),
            ):
                if delzero:
                    nullidx = (
                        temp_physdata[self.outputcolnames[recordingtype]].isna()
                    ) | (temp_physdata[self.outputcolnames[recordingtype]] == 0)
                    delreason = "NULL or Zero"
                else:
                    nullidx = temp_physdata[self.outputcolnames[recordingtype]].isna()
                    delreason = "NULL"
            else:
                nullidx = temp_physdata[self.outputcolnames[recordingtype]].isna()
                delreason = "NULL"
            if temp_physdata[nullidx].shape[0] > 0:
                to_delete = temp_physdata[nullidx]
                to_delete["Reason"] = f"{delreason} Measurement"
                self.brphysdata_deleted = pd.concat(
                    [self.brphysdata_deleted, to_delete], ignore_index=True
                )
            self.brphysdata = pd.concat(
                [self.brphysdata, temp_physdata[~nullidx]], ignore_index=True
            )

    def generateBreathePhysdataTableFromMeasureTables(self):
        """Iterate through the data and generate the breathe physdata table."""
        breatheRowsToAdd = [
            # ["Activity", "CalorieRecording", 1],
            # ["Coughing", "CoughRecording", 1],
            # ["HeartRate", "RestingHRRecording", 1],
            ["Oximeter", "O2SaturationRecording", 1],
            ["Oximeter", "PulseRateRecording", 1],
            # ["Sleep", "MinsAsleepRecording", 0],
            # ["Sleep", "MinsAwakeRecording", 0],
            ["Spirometer", "FEV1Recording", 1],
            ["Spirometer", "FEF2575Recording", 1],
            ["Spirometer", "PEFRecording", 1],
            ["Spirometer", "FEV075Recording", 1],
            ["Spirometer", "FEV1DivFEV6Recording", 1],
            ["Spirometer", "FEV6Recording", 1],
            # ["Temperature", "TemperatureRecording", 1],
            # ["Weight", "WeightRecording", 1],
            # ["Wellbeing", "WellnessRecording", 1],
            # ["Wellbeing", "HasColdOrFluRecording", 0],
            # ["Wellbeing", "HasHayFeverRecording", 0],
        ]
        for i in breatheRowsToAdd:
            self.addBreatheRowsForMeasure(self.measure_tables[i[0]], i[1], i[2])
        # temp_rows = self.brphysdata[self.brphysdata["RecordingType"] == "FEV1Recording"]

        # merge FEV1 Recording with the Redcap FEV1 data and calc a percentage of a baseline
        # temp_rows = temp_rows.merge(
        #     self.brPatient.filter(["ID", "CalcPredictedFEV1"]).rename(
        #         columns={"ID": "SmartCareID"}
        #     ),
        #     how="left",
        #     left_on="SmartCareID",
        #     right_on="SmartCareID",
        # )
        # temp_rows["RecordingType"] = "LungFunctionRecording"
        # temp_rows["CalcFEV1_"] = (100 * temp_rows["FEV"]) / temp_rows[
        #     "CalcPredictedFEV1"
        # ]
        # temp_rows["FEV"] = 0
        # temp_rows = temp_rows.drop(["CalcPredictedFEV1"], axis=1)
        # self.brphysdata = pd.concat([self.brphysdata, temp_rows], ignore_index=True)

    def findAndDeleteAnomalousMeasures(self, recordingtype, lowerthresh, upperthresh):
        """Function to trim data to within thresholds and add trimmed data to the deleted table

        Args:
            recordingtype (str): recording being trimmed
            lowerthresh (float): lower threshold
            upperthresh (vloat): upper threshold
        """
        outputcolname = self.outputcolnames[recordingtype]
        to_delete_idx = (self.brphysdata["RecordingType"] == recordingtype) & (
            (self.brphysdata[outputcolname] < lowerthresh)
            | (self.brphysdata[outputcolname] > upperthresh)
        )
        to_delete = self.brphysdata[to_delete_idx]
        to_delete["Reason"] = "Anomalous Value"
        self.brphysdata_deleted = pd.concat(
            [self.brphysdata_deleted, to_delete], ignore_index=True
        )
        self.brphysdata = self.brphysdata[~to_delete_idx]

    def demographicFunction(self, valueArray):
        """Generate summary statistics for data array

        Args:
            valueArray (_type_): _description_

        Returns:
            _type_: _description_
        """
        valueArray = [i for i in valueArray if str(i) != "nan"]
        if valueArray == []:
            return np.nan
        else:
            valueArray = np.sort(np.asarray(valueArray))
            mid50 = valueArray[
                round(len(valueArray) * 0.25) : round(len(valueArray) * 0.75)
            ]
            xb25 = valueArray[round(len(valueArray) * 0.25) :]
            xu25 = valueArray[: round(len(valueArray) * 0.75)]
            return {
                "mean": np.nanmean(valueArray),
                "std": np.nanstd(valueArray),
                "mini": min(valueArray),
                "maxi": max(valueArray),
                "mid50mean": np.nanmean(mid50),
                "mid50std": np.nanstd(mid50),
                "mid50min": min(mid50),
                "mid50max": max(mid50),
                "xb25mean": np.nanmean(xb25),
                "xb25std": np.nanstd(xb25),
                "xb25min": min(xb25),
                "xb25max": max(xb25),
                "xu25mean": np.nanmean(xu25),
                "xu25std": np.nanstd(xu25),
                "xu25min": min(xu25),
                "xu25max": max(xu25),
            }

    def generateDataDemographicsTables(self):
        """Summarise data for each patient and recording type"""

        temp_physdata = self.brphysdata
        temp_physdata = temp_physdata.drop(
            [
                "UserName",
                "ScaledDateNum",
                "DateNum",
                "Date_TimeRecorded",
                "CaptureType",
            ],
            axis=1,
        )

        ##summarise at a patient/recording type level
        # TODO turn the below into a function as it is used multiple times
        demographicstable = temp_physdata.groupby(
            ["SmartCareID", "RecordingType"], as_index=False
        ).agg(lambda x: self.demographicFunction(list(x)))

        values = [
            i[self.outputcolnames[i["RecordingType"]]]
            for i in demographicstable.to_dict("records")
        ]

        for i in [
            "mean",
            "std",
            "mini",
            "maxi",
            "mid50mean",
            "mid50std",
            "mid50min",
            "mid50max",
            "xb25mean",
            "xb25std",
            "xb25min",
            "xb25max",
            "xu25mean",
            "xu25std",
            "xu25min",
            "xu25max",
        ]:

            def tmp_fn(j):
                if isinstance(j, dict):
                    return j[i]
                else:
                    return np.nan

            demographicstable[i] = [tmp_fn(j) for j in values]

        demographicstable = demographicstable.drop(
            [
                "FEV",
                "WeightInKg",
                "O2Saturation",
                "Pulse_BPM_",
                "Temp_degC_",
                "Calories",
                "Rating",
                "Sleep",
                "HasCondition",
                # "CalcFEV1_",
            ],
            axis=1,
        )

        # count records for eadch patient/recording type combo
        demographicstable_count = (
            temp_physdata.groupby(["SmartCareID", "RecordingType"], as_index=False)
            .size()
            .rename(columns={"size": "GroupCount"})
        )

        demographicstable = demographicstable.merge(
            demographicstable_count,
            how="left",
            left_on=["SmartCareID", "RecordingType"],
            right_on=["SmartCareID", "RecordingType"],
        )

        temp_physdata = temp_physdata.drop(["SmartCareID"], axis=1)

        # find overall summaries of entire cohort
        overalltable = temp_physdata.groupby(["RecordingType"], as_index=False).agg(
            lambda x: self.demographicFunction(list(x))
        )

        values = [
            i[self.outputcolnames[i["RecordingType"]]]
            for i in overalltable.to_dict("records")
        ]
        for i in [
            "mean",
            "std",
            "mini",
            "maxi",
            "mid50mean",
            "mid50std",
            "mid50min",
            "mid50max",
            "xb25mean",
            "xb25std",
            "xb25min",
            "xb25max",
            "xu25mean",
            "xu25std",
            "xu25min",
            "xu25max",
        ]:

            overalltable[i] = [tmp_fn(j) for j in values]

        overalltable = overalltable.drop(
            [
                "FEV",
                "WeightInKg",
                "O2Saturation",
                "Pulse_BPM_",
                "Temp_degC_",
                "Calories",
                "Rating",
                "Sleep",
                "HasCondition",
                # "CalcFEV1_",
            ],
            axis=1,
        )
        overalltable_count = (
            temp_physdata.groupby(["RecordingType"], as_index=False)
            .size()
            .rename(columns={"size": "GroupCount"})
        )
        overalltable = overalltable.merge(
            overalltable_count,
            how="left",
            left_on="RecordingType",
            right_on="RecordingType",
        )
        measurecounttable = demographicstable.filter(
            ["SmartCareID", "RecordingType", "GroupCount"]
        )

        today = datetime.today()
        out_dict = {
            "MeasureCountByPatient": measurecounttable,
            "DataDemographicsByPatient": demographicstable,
            "OverallDataDemographics": overalltable,
        }
        # tmp_df = pd.DataFrame(data=out_dict, index=[0])

        # tmp_df.to_excel(
        #     f"{dh.get_path_to_main()}/ExcelFiles/BR/BRDataDemographicsByPatient-{today.strftime('%Y%m%d')}T{today.strftime('%H%M%S')}.xlsx",
        # )

    def handleBreatheDuplicateMeasuresOneSheet(self):
        temp_physdata = (
            self.brphysdata.sort_values(
                by=["SmartCareID", "RecordingType", "Date_TimeRecorded", "Sleep"],
                ascending=True,
            )
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata["ToRemove"] = False
        dict_temp_physdata = temp_physdata.to_dict("records")

        for i in range(temp_physdata.shape[0] - 1):
            curr_recording_type = dict_temp_physdata[i]["RecordingType"]
            next_recording_type = dict_temp_physdata[i + 1]["RecordingType"]
            # Check if current and next row have the same SmartCareID and RecordingType
            if (
                dict_temp_physdata[i]["SmartCareID"]
                == dict_temp_physdata[i + 1]["SmartCareID"]
                and curr_recording_type == next_recording_type
                and curr_recording_type in ["MinsAsleepRecording", "MinsAwakeRecording"]
                and dict_temp_physdata[i]["Date_TimeRecorded"]
                == dict_temp_physdata[i + 1]["Date_TimeRecorded"]
                and dict_temp_physdata[i]["Sleep"] == dict_temp_physdata[i + 1]["Sleep"]
            ):
                # Check if the current and next rows are identical 'Sleep' rows
                # For identical 'Sleep' rows the first row is marked for deletion
                temp_physdata.at[i, "ToRemove"] = True

        temp_physdata = (
            temp_physdata[temp_physdata["ToRemove"] == False]
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata = (
            temp_physdata.sort_values(
                by=["SmartCareID", "RecordingType", "Date_TimeRecorded", "Sleep"],
                ascending=True,
            )
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata["ToRemove"] = False
        dict_temp_physdata = temp_physdata.to_dict("records")

        for i in range(temp_physdata.shape[0] - 1):
            curr_recording_type = temp_physdata.iloc[i]["RecordingType"]
            next_recording_type = temp_physdata.iloc[i + 1]["RecordingType"]
            # Check if current and next row have the same SmartCareID and RecordingType and Date_TimeRecorded
            if (
                temp_physdata.iloc[i]["SmartCareID"]
                == temp_physdata.iloc[i + 1]["SmartCareID"]
                and curr_recording_type == next_recording_type
                and temp_physdata.iloc[i]["Date_TimeRecorded"]
                == temp_physdata.iloc[i + 1]["Date_TimeRecorded"]
            ):
                # Check if current row is MinsAsleepRecording
                if curr_recording_type == "MinsAsleepRecording":
                    # Update the next row with the sum of all the MinsAsleepRecording values
                    temp_physdata.at[i + 1, "Sleep"] += temp_physdata.iloc[i]["Sleep"]
                    temp_physdata.at[i, "ToRemove"] = True
                elif curr_recording_type in ["MinsAwakeRecording"]:
                    temp_physdata.at[i, "ToRemove"] = True
                    temp_physdata.at[
                        i + 1, self.outputcolnames[next_recording_type]
                    ] += dict_temp_physdata[i][self.outputcolnames[curr_recording_type]]
                else:
                    temp_physdata.at[i, "ToRemove"] = True
                    temp_physdata.at[
                        i + 1, self.outputcolnames[next_recording_type]
                    ] = max(
                        dict_temp_physdata[i][self.outputcolnames[curr_recording_type]],
                        dict_temp_physdata[i + 1][
                            self.outputcolnames[next_recording_type]
                        ],
                    )

        temp_physdata = (
            temp_physdata[temp_physdata["ToRemove"] == False]
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata = (
            temp_physdata.sort_values(
                by=["SmartCareID", "RecordingType", "Date_TimeRecorded"], ascending=True
            )
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata["ToRemove"] = False
        dict_temp_physdata = temp_physdata.to_dict("records")

        for i in range(temp_physdata.shape[0] - 1):
            curr_recording_type = dict_temp_physdata[i]["RecordingType"]
            next_recording_type = dict_temp_physdata[i + 1]["RecordingType"]
            # Check if current and next row have the same SmartCareID and RecordingType
            if (
                dict_temp_physdata[i]["SmartCareID"]
                == dict_temp_physdata[i + 1]["SmartCareID"]
                and curr_recording_type == next_recording_type
                and abs(
                    (
                        dict_temp_physdata[i]["Date_TimeRecorded"]
                        - dict_temp_physdata[i + 1]["Date_TimeRecorded"]
                    ).total_seconds()
                    / 60
                )
                < 60
            ):
                # Check if the difference in time between the current and next rows is less than 60 minutes
                # For similar rows the first row is marked for deletion and the second row's value is updated
                temp_physdata.at[i, "ToRemove"] = True
                if curr_recording_type in ["MinsAsleepRecording", "MinsAwakeRecording"]:
                    temp_physdata.at[
                        i + 1, self.outputcolnames[next_recording_type]
                    ] += dict_temp_physdata[i][self.outputcolnames[curr_recording_type]]
                else:
                    temp_physdata.at[
                        i + 1, self.outputcolnames[next_recording_type]
                    ] = max(
                        temp_physdata.at[
                            i + 1, self.outputcolnames[next_recording_type]
                        ],
                        temp_physdata.at[i, self.outputcolnames[curr_recording_type]],
                    )
                    temp_physdata.at[i, "ToRemove"] = True

        temp_physdata = (
            temp_physdata[temp_physdata["ToRemove"] == False]
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata = (
            temp_physdata.sort_values(
                by=["SmartCareID", "RecordingType", "Date_TimeRecorded"], ascending=True
            )
            .reset_index()
            .drop(columns=["index"])
        )
        temp_physdata["ToRemove"] = False
        dict_temp_physdata = temp_physdata.to_dict("records")

        for i in range(temp_physdata.shape[0] - 1):
            curr_recording_type = dict_temp_physdata[i]["RecordingType"]
            next_recording_type = dict_temp_physdata[i + 1]["RecordingType"]
            # Check if current and next row have the same SmartCareID and RecordingType
            if (
                dict_temp_physdata[i]["SmartCareID"]
                == dict_temp_physdata[i + 1]["SmartCareID"]
                and curr_recording_type == next_recording_type
                and dict_temp_physdata[i]["DateNum"]
                == dict_temp_physdata[i + 1]["DateNum"]
            ):
                # Find the number of consecutive rows with the same SmartCareID, RecordingType, and DateNum
                n_consecutive_rows = 2
                for j in range(i + 2, temp_physdata.shape[0]):
                    if (
                        dict_temp_physdata[j]["SmartCareID"]
                        == dict_temp_physdata[i]["SmartCareID"]
                        and dict_temp_physdata[j]["RecordingType"]
                        == curr_recording_type
                        and dict_temp_physdata[j]["DateNum"]
                        == dict_temp_physdata[i]["DateNum"]
                    ):
                        n_consecutive_rows += 1
                    else:
                        break
                # For rows on the same day the first row is marked for deletion
                for j in range(i, i + n_consecutive_rows - 1):
                    temp_physdata.at[j, "ToRemove"] = True
                # For 'Sleep' rows the consecutive rows' value is updated with the mean of all values
                if curr_recording_type in ["MinsAsleepRecording", "MinsAwakeRecording"]:
                    # For 'Sleep' rows the second row's sleep value is updated with the sum of the sleep values
                    temp_physdata.at[
                        i + 1, self.outputcolnames[next_recording_type]
                    ] += temp_physdata.at[i, self.outputcolnames[curr_recording_type]]
                else:
                    mean_value = temp_physdata.loc[
                        i : i + n_consecutive_rows - 1,
                        self.outputcolnames[curr_recording_type],
                    ].mean()
                    indexer = temp_physdata.index[i : i + n_consecutive_rows - 1]
                    temp_physdata.loc[
                        indexer, self.outputcolnames[curr_recording_type]
                    ] = mean_value

        # Remove the rows marked for deletion
        temp_physdata = (
            temp_physdata[temp_physdata["ToRemove"] == False]
            .reset_index()
            .drop(columns=["index"])
        )
        self.brphysdata = temp_physdata.drop(columns=["ToRemove"])
        today = datetime.today()
        self.brphysdata.to_csv(
            f"{dh.get_path_to_main()}/ExcelFiles/BR/BRDuplicates-{today.strftime('%Y%m%d')}T{today.strftime('%H%M%S')}.csv",
        )

    def runPreprocess(self):
        self.loadMeasureTables()
        self.generateBreathePhysdataTableFromMeasureTables()
        self.brphysdata_original = self.brphysdata
        self.broffset = (
            min(self.brphysdata["Date_TimeRecorded"]).date() - datetime(1, 1, 1).date()
        ).days
        self.brphysdata["DateNum"] = self.brphysdata["Date_TimeRecorded"].apply(
            lambda x: math.ceil(
                (x.date() - datetime(1, 1, 1).date()).days - self.broffset
            )
        )

        measureRanges = [
            [
                "CalorieRecording",
                -1,
                6000,
            ],  # the way it is set up allows for calories of -1 to be used
            ["FEV1Recording", 0.1, 6],
            ["FEV6Recording", 0.2, 7],
            ["O2SaturationRecording", 70, 100],
            ["PulseRateRecording", 40, 200],
            ["RestingHRRecording", 40, 120],
            ["MinsAsleepRecording", -1, 1200],  # can be asleep for 20 hours??
            [
                "MinsAwakeRecording",
                -1,
                600,
            ],  # MinsAwakeRecording this is minutes awake during the night (mins not sleeping)
            ["TemperatureRecording", 34, 40],
            ["WeightRecording", 30, 120],
        ]
        for i in measureRanges:
            self.findAndDeleteAnomalousMeasures(i[0], i[1], i[2])
        self.brphysdata_predupehandling = self.brphysdata

        # self.generateDataDemographicsTables() # done twice, unsure why

        self.handleBreatheDuplicateMeasuresOneSheet()

        minDatesByPatient = (
            self.brphysdata.filter(["SmartCareID", "DateNum"])
            .groupby("SmartCareID", as_index=False)
            .min()
        )
        minDatesByPatient = minDatesByPatient.rename(
            columns={"DateNum": "MinPatientDateNum"}
        )
        self.brphysdata = self.brphysdata.merge(
            minDatesByPatient, how="left", left_on="SmartCareID", right_on="SmartCareID"
        )
        self.brphysdata["ScaledDateNum"] = (
            self.brphysdata["DateNum"] - self.brphysdata["MinPatientDateNum"] + 1
        )
        self.brphysdata = self.brphysdata.drop(columns=["MinPatientDateNum"])

        self.brphysdata_predateoutlierhandling = (
            self.brphysdata
        )  # will be the same for BR as no outlier handling occurs

        # analyseAndHandleDateOutliers() # not done for BR
        # createMeasuresHeatmapWithStudyPeriod() # Makes plots so havent done

        # TODO: removed permanently on 02.04.2024 due to code error, will fix later
        # self.generateDataDemographicsTables()  # second time run

    def returnData(self):
        return (
            self.brphysdata,
            self.broffset,
            self.brphysdata_deleted,
            self.brphysdata_original,
            self.brphysdata_predupehandling,
            self.brphysdata_predateoutlierhandling,
        )

    def exportData(self):
        today = datetime.today()
        self.brphysdata.to_excel(
            f"{dh.get_path_to_main()}/ExcelFiles/BR/BRPhysdata-{today.strftime('%Y%m%d')}T{today.strftime('%H%M%S')}.xlsx",
            index=False,
        )
        self.brphysdata_deleted[
            self.brphysdata_deleted["Reason"] != "NULL Measurement"
        ].to_excel(
            f"{dh.get_path_to_main()}/ExcelFiles/BR/BRDeletedMeasurementData-{today.strftime('%Y%m%d')}T{today.strftime('%H%M%S')}.xlsx",
            index=False,
        )
        pd.DataFrame.from_dict({"Offset": [self.broffset]}).to_excel(
            f"{dh.get_path_to_main()}/ExcelFiles/BR/BROffset-{today.strftime('%Y%m%d')}T{today.strftime('%H%M%S')}.xlsx",
            index=False,
        )


if __name__ == "__main__":
    a = datetime.now()
    MD = MeasurementData()
    MD.runPreprocess()
    MD.exportData()
    """physdata, _, _, _, _, _ = MD.returnData()
    print(physdata)"""
    print("final " + str(datetime.now() - a))
