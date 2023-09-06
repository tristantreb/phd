from datetime import timedelta

import pandas as pd

datadir = "../../../../SmartCareData/"

# This document contains two ways to compute exacerbation labels, which are exacerbated, not exacerbated, in recovery
# 1. Is with the results of the predictive classifier from the work done by Damian
# 2. (proved less useful than 1.) Is using a rule of thumb for when the patients are in which state. We define:
#     - Exacerbated from 1 week days before treatment start, excluding treatment start day
#     - Not exacerbated as until 3 weeks before treatment start
#     - Recovery: during treatment
#     - Between 3 and 1 week before treatment start is thrown away.

one_day = timedelta(days=1)


# Merge exacerbation labels from the predictive classifier to O2_FEV1
def inner_merge_with(O2_FEV1, pred_ex_labels):
    O2_FEV1_out = O2_FEV1.merge(
        pred_ex_labels["Is Exacerbated"],
        how="inner",
        on=["ID", "Date recorded"],
        validate="1:1",
    )
    print(
        "** Inner merge of O2_FEV1 and exacerbated labels on 'ID' and 'Date recorded' **\nData has now {} entries and {} IDs (initially {} & {} in O2_FEV1, {} in pred_ex_labels)".format(
            O2_FEV1_out.shape[0],
            O2_FEV1_out.ID.nunique(),
            O2_FEV1.shape[0],
            O2_FEV1.ID.nunique(),
            pred_ex_labels.shape[0],
        )
    )
    return O2_FEV1_out


# METHOD 1: getting the exacerbation labels inferred with the predictive classifier
# Exclude no ex allows to exclude individuals don't have a measurement in an exacerbated period
def load(exclude_no_ex=False):
    # Get exacerbation labels from the predictive classifier
    pred_ex_labels = pd.read_csv(datadir + "pmFeatureIIndex.csv")

    pred_ex_labels["Is Exacerbated"] = pd.read_csv(
        datadir + "pmExABxElLabels.csv", dtype=bool
    )

    # Data types transformation. Use datetime.date for Date recorded and string for ID
    pred_ex_labels["Date recorded"] = pd.to_datetime(pred_ex_labels["CalcDate"]).dt.date
    pred_ex_labels["ID"] = pred_ex_labels["ID"].astype(str)

    if exclude_no_ex:
        ids_ex = pred_ex_labels[pred_ex_labels["Is Exacerbated"] == True].ID.unique()
        print(
            f"Keeping {len(ids_ex)}/{len(pred_ex_labels.ID.unique())} individuals that have a measurement during an exacerbated period (Is Exacerbated == true at least once)"
        )
        pred_ex_labels = pred_ex_labels[pred_ex_labels.ID.isin(ids_ex)]

    # Set Multi index to prepare the merge with O2_FEV1
    pred_ex_labels = pred_ex_labels.set_index(["ID", "Date recorded"])

    # Removing NaN values in Is Exacerbated
    pred_ex_labels = pred_ex_labels[~pred_ex_labels["Is Exacerbated"].isna()]

    # Record the number of rows
    print(
        "Exacerbated labels data from the predictive classifier has {} entries ({} exacerbated, {} not "
        "exacerbated measurements, {} NaN)".format(
            pred_ex_labels.shape[0],
            pred_ex_labels[pred_ex_labels["Is Exacerbated"] == True].shape[0],
            pred_ex_labels[pred_ex_labels["Is Exacerbated"] == False].shape[0],
            pred_ex_labels[pred_ex_labels["Is Exacerbated"].isna()].shape[0],
        )
    )
    return pred_ex_labels


# Drops individuals that have no measurement in an exacerbated period
def drop_where_no_ex(df):
    ids_ex = df[df["Is Exacerbated"] == True].ID.unique()
    print(
        f"Keeping {len(ids_ex)}/{len(df.ID.unique())} individuals that have a measurement in exacerbated period"
    )
    return df[df.ID.isin(ids_ex)]


# METHOD 2: using rule of thumbs around treatment start/end \dates
# to compute exacerbation labels
def compute_ex_labels_from_heuristics(antibioticsdata, patientsdata, O2_FEV1):
    for id in O2_FEV1.ID.unique():
        patient_antibioticsdata = _get_rows_for_id(id, antibioticsdata)
        ex_labels = _get_patient_ex_labels(
            patient_antibioticsdata,
            patientsdata,
            numdays_before_ab_start_is_exacerbated=7,
            numdays_before_ab_start_not_exacerbated=21,
        )
        O2_FEV1["Exacerbation Labels"] = O2_FEV1["Date recorded"].apply(
            lambda x: add_measurement_exacerbation_label(x, ex_labels)
        )
    return O2_FEV1


def _get_rows_for_id(id, data):
    return data[data.ID == id].reset_index()


# The patient status can be exacerbated, not exacerbated, in recovery
def _get_patient_ex_labels(
    patient_antibioticsdata,
    patientsdata,
    numdays_before_ab_start_is_exacerbated=7,
    numdays_before_ab_start_not_exacerbated=21,
):
    recovery_period_agg = []
    exacerbated_period_agg = []
    not_exacerbated_period_agg = []

    # Compute the periods for each antibiotic treatment and aggregates periods with same status
    for i in range(0, len(patient_antibioticsdata)):
        study_start = _get_patient_study_start(patient_antibioticsdata, patientsdata)
        antibiotic_start = patient_antibioticsdata["Start Date"][i]
        antibiotic_end = patient_antibioticsdata["Stop Date"][i]
        if i > 0:
            prev_antibiotic_end = patient_antibioticsdata["Stop Date"][i - 1]

        recovery_period_agg.extend(_create_date_range(antibiotic_start, antibiotic_end))

        # If it's the first ab on the list then the individual can't be labelled as exacerbated before the study
        # start date (included)
        if i == 0:
            # Adding - 1 to patient_study_start to include it, see function
            is_exacerbated_start = _get_is_exacerbated_start(
                antibiotic_start,
                study_start - one_day,
                numdays_before_ab_start_is_exacerbated,
            )
        else:
            is_exacerbated_start = _get_is_exacerbated_start(
                antibiotic_start,
                prev_antibiotic_end,
                numdays_before_ab_start_is_exacerbated,
            )
        exacerbated_period_agg.extend(
            _create_date_range(
                is_exacerbated_start, _get_is_exacerbated_end(antibiotic_end)
            )
        )

        # If it's the first ab on the list then the person is has not been exacerbated since the beginning of the study
        if i == 0:
            not_exacerbated_start = _get_not_exacerbated_start(study_start - one_day)
            not_exacerbated_end = _get_not_exacerbated_end(
                antibiotic_start,
                study_start - one_day,
                numdays_before_ab_start_not_exacerbated,
            )
        else:
            not_exacerbated_start = _get_not_exacerbated_start(prev_antibiotic_end)
            not_exacerbated_end = _get_not_exacerbated_end(
                antibiotic_start,
                prev_antibiotic_end,
                numdays_before_ab_start_not_exacerbated,
            )

        not_exacerbated_period_agg.extend(
            _create_date_range(not_exacerbated_start, not_exacerbated_end)
        )

    return {
        "recovery_period": recovery_period_agg,
        "exacerbated_period": exacerbated_period_agg,
        "not_exacerbated_period": not_exacerbated_period_agg,
    }


def _get_patient_study_start(patient_antibioticsdata, patientsdata):
    id = patient_antibioticsdata.ID.unique()[0]
    study_start = get_rows_for_id(id, patientsdata)["Study Date"][0]
    return study_start


# Patient status period definition
# 1. The recovery period is the treatment period


# 2. The exacerbated period is from a week before treatment start to treatment start excluded
def _get_is_exacerbated_start(
    antibiotic_start, prev_antibiotics_end, numdays_before_start=7
):
    # The patient cannot be labelled as exacerbated before the end of the previous antibiotic (or on same day)
    return max(
        prev_antibiotics_end + one_day,
        antibiotic_start - timedelta(days=numdays_before_start),
    )


def _get_is_exacerbated_end(antibiotic_start):
    return antibiotic_start - one_day


# 3. The not exacerbated is from the previous antibiotics'end (excluded) to three weeks before treatment start (
# excluded)
def _get_not_exacerbated_start(prev_antibiotics_end):
    return prev_antibiotics_end + one_day


def _get_not_exacerbated_end(
    antibiotic_start, prev_antibiotics_end, numdays_before_start=21
):
    # The patient cannot be labelled as not exacerbated before the end of the previous antibiotic
    return max(
        prev_antibiotics_end + one_day,
        antibiotic_start - timedelta(days=numdays_before_start),
    )


# Returns a list of datetimes for each day between start and end, if applicable
def _create_date_range(start, end):
    if start < end:
        delta = end - start
        return [start + timedelta(days=x) for x in range(delta.days)]
    # start >= end, which does not make sense
    else:
        return []


# Applies the right label to the measurement
def add_measurement_exacerbation_label(measurement_date, exacerbation_labels):
    recovery_days = exacerbation_labels["recovery_period"]
    exacerbated_days = exacerbation_labels["exacerbated_period"]
    stable_days = exacerbation_labels["not_exacerbated_period"]

    if measurement_date in recovery_days:
        return "Recovery Period"
    elif measurement_date in exacerbated_days:
        return "Exacerbation Period"
    elif measurement_date in stable_days:
        return "Stable Period"
    else:
        return "Undefined Period"
