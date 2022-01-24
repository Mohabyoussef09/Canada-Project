import numpy as np


def update_invalid_vars_list(data,probabilityLabel):
    """Flag non-model variables: categorical variables, variables with missing values."""
    probability_label=probabilityLabel
    categorical_list = data.select_dtypes(include="object").columns.tolist()
    missing_list = [x for x in data.columns.tolist() if x not in
                    [probability_label] + categorical_list +
                    data.dropna(axis=1).columns.tolist()]
    invalid_vars = list(np.sort(data.columns.get_indexer(categorical_list
                                                                      + missing_list)))
    # TODO #421: Find a way to serialize int64 and avoid line below.
    invalid_vars = [int(x) for x in invalid_vars]

    #excluded_vars = self.invalid_vars.copy()
    return invalid_vars

def non_event_labels(data,probabilityLabel,targetLabel):
        non_events = ([probabilityLabel]  if targetLabel is None else [probabilityLabel, targetLabel] )
        return [x for x in data.columns if x not in non_events]

def model_qualified_var_labels(data,non_event_labels,invalid_vars):
    if non_event_labels:
        return [
            x
            for x in non_event_labels
            if x not in data.columns[invalid_vars]
        ]