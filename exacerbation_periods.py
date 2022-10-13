from datetime import timedelta

one_day = timedelta(days=1)


def list_patients(data):
    return data.ID.unique()


def get_rows_for_id(id, data):
    return data[data.ID == id].reset_index()


def get_patient_study_start(patient_antibioticsdata, patientsdata):
    id = patient_antibioticsdata.ID.unique()[0]
    study_start = get_rows_for_id(id, patientsdata)['Study Date'][0]
    return study_start


# Patient status period definition
# 1. The recovery period is the treatment period

# 2. The exacerbated period is from a week before treatment start to treatment start excluded
def get_is_exacerbated_start(antibiotic_start, prev_antibiotics_end):
    # The patient cannot be labelled as exacerbated before the end of the previous antibiotic (or on same day)
    return max(prev_antibiotics_end + one_day, antibiotic_start - timedelta(days=7))


def get_is_exacerbated_end(antibiotic_start):
    return antibiotic_start - one_day


# 3. The not exacerbated is from the previous antibiotics'end (excluded) to three weeks before treatment start (
# excluded)
def get_not_exacerbated_start(prev_antibiotics_end):
    return prev_antibiotics_end + one_day


def get_not_exacerbated_end(antibiotic_start, prev_antibiotics_end):
    # The patient cannot be labelled as not exacerbated before the end of the previous antibiotic
    return max(prev_antibiotics_end + one_day, antibiotic_start - timedelta(days=21))


# The patient status can be exacerbated, not exacerbated, in recovery
def get_patient_exacerbation_labels(patient_antibioticsdata, patientsdata):
    recovery_period_agg = []
    exacerbated_period_agg = []
    not_exacerbated_period_agg = []

    # Compute the periods for each antibiotic treatment and aggregates periods with same status
    for i in range(0, len(patient_antibioticsdata)):
        study_start = get_patient_study_start(patient_antibioticsdata, patientsdata)
        antibiotic_start = patient_antibioticsdata['Start Date'][i]
        antibiotic_end = patient_antibioticsdata['Stop Date'][i]
        if i > 0: prev_antibiotic_end = patient_antibioticsdata['Stop Date'][i - 1]

        recovery_period_agg.extend(create_date_range(antibiotic_start, antibiotic_end))

        # If it's the first ab on the list then the individual can't be labelled as exacerbated before the study
        # start date (included)
        if i == 0:
            # Adding - 1 to patient_study_start to include it, see function
            is_exacerbated_start = get_is_exacerbated_start(antibiotic_start, study_start - one_day)
        else:
            is_exacerbated_start = get_is_exacerbated_start(antibiotic_start, prev_antibiotic_end)
        exacerbated_period_agg.extend(create_date_range(is_exacerbated_start, get_is_exacerbated_end(antibiotic_end)))

        # If it's the first ab on the list then the person is has not been exacerbated since the beginning of the study
        if i == 0:
            not_exacerbated_start = get_not_exacerbated_start(study_start - one_day)
            not_exacerbated_end = get_not_exacerbated_end(antibiotic_start, study_start - one_day)
        else:
            not_exacerbated_start = get_not_exacerbated_start(prev_antibiotic_end)
            not_exacerbated_end = get_not_exacerbated_end(antibiotic_start, prev_antibiotic_end)

        not_exacerbated_period_agg.extend(create_date_range(not_exacerbated_start, not_exacerbated_end))

    return {"recovery_period": recovery_period_agg,
            "exacerbated_period": exacerbated_period_agg,
            "not_exacerbated_period": not_exacerbated_period_agg}


# Returns a list of datetimes for each day between start and end, if applicable
def create_date_range(start, end):
    if start < end:
        delta = end - start
        return [start + timedelta(days=x) for x in range(delta.days)]
    # start >= end, which does not make sense
    else:
        return []


# Applies the right label to the measurement
def add_measurement_exacerbation_label(measurement_date, exacerbation_labels):
    recovery_days = exacerbation_labels['recovery_period']
    exacerbated_days = exacerbation_labels['exacerbated_period']
    stable_days = exacerbation_labels['not_exacerbated_period']

    if measurement_date in recovery_days:
        return "Recovery Period"
    elif measurement_date in exacerbated_days:
        return "Exacerbation Period"
    elif measurement_date in stable_days:
        return "Stable Period"
    else:
        return "Undefined Period"
