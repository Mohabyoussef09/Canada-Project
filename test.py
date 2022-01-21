import df_utils
from binning_calculations import calculate_data_variables,calculate_bin_variables,apply_binning
from analyze_calculations import calculate_distribution_variables,calculate_volatility_variables,calculate_correlation_variables
from dataset import Dataset
from main_window import MainWindow
from train_settings import IncludeExclude
from analyze_display import AnalyzeDisplay
import pandas as pd
from binning_settings import (
    BinAsMode,
    BinningNumericMode,
    BinningSettings,
)
from modellicity.src.modellicity.settings import settings
from train_settings import ModelType, TrainSettings
import config
import numpy as np
##########B in page code###########
from train_calculations import TrainCalculate

b=BinningSettings()
data=pd.read_csv("train.csv")
d=calculate_data_variables(data,b)
d.head()

target="default"
bin_vars=calculate_bin_variables(data,b,target,d)
print(b)

parent_list = list(bin_vars["variable_parent"].unique())
binned_list = list(bin_vars["variable"].unique())
df_binned = apply_binning(data[parent_list], bin_vars)[binned_list]

#merge data dataframe with df_binned dataframe when user press ok
datasetObject=Dataset()
finalBinningData=datasetObject.add_binned_feature_vars(data,df_binned)
print(b)

#####################################

################ Start analyze tab ######################
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
analyze=AnalyzeDisplay()
invalid_vars= list(range(config.DEFAULT_COL_COUNT))

probabilityLabel="probability"
targetLabel="default"
invalid_vars=update_invalid_vars_list(data,probabilityLabel)
non_event_labels=non_event_labels(data,probabilityLabel,targetLabel)
modelQualifiedVarLabels=model_qualified_var_labels(data,non_event_labels,invalid_vars)

distributionVariables=calculate_distribution_variables(data[modelQualifiedVarLabels])
volatilityVariables=calculate_volatility_variables(data[modelQualifiedVarLabels])
correlatioVariables=calculate_correlation_variables(data[modelQualifiedVarLabels])


#when user press ok

def update_excluded_vars_list(data, user_excluded_vars: list[int]):
        if ( min(user_excluded_vars) < 0  or max(user_excluded_vars) >= data.shape[1]):
            return sorted(list(set(invalid_vars).union(set(user_excluded_vars))))



#user_excluded_var_indices = [i for i, x in enumerate(list(data.columns)) if x in invalid_vars]
analyzeOkPressedData=finalBinningData#update_excluded_vars_list(data,user_excluded_var_indices)
x=10


############################################
#train screen
###########################################


def model_qualified_var_labels(data):
    if non_event_labels:
        return [
            x
            for x in non_event_labels
            if x not in data.columns[invalid_vars]
        ]

excluded_vars=invalid_vars
def calculate_disabled_var_labels_from_dataset(data):
    all_vars = list(data.columns)
    var_names = model_qualified_var_labels(data)
    disabled_var_labels = []
    for i in excluded_vars:
        var = all_vars[i]
        if var in var_names:
            disabled_var_labels.append(var)
    return disabled_var_labels


def calculate_variable_selection_panel_vars(data):
    disabled_var_labels = calculate_disabled_var_labels_from_dataset(data)
    var_list = model_qualified_var_labels(data)
    include_exclude_list = []
    manually_set_list = [False] * len(var_list)
    for var in var_list:
        if var in disabled_var_labels:
            include_exclude_list.append(IncludeExclude.EXCLUDE)
        else:
            include_exclude_list.append(IncludeExclude.INCLUDE)

    return pd.DataFrame({"variable": var_list,
                         "include_exclude_panel": include_exclude_list,
                         "include_exclude": include_exclude_list,
                         "manually_set": manually_set_list})

def calculate_num_obs(data):
    if data is None:
        return None
    return data.shape[0]

def calculate_num_features(disabled_var_labels):
    return len(modelQualifiedVarLabels) - len(disabled_var_labels)

def calculate_num_parameters(disabled_var_labels):
    ''''
    if settings.model_settings.model_framework == ModelType.LOGISTIC_REGRESSION:
        return calculate_num_features(disabled_var_labels) + 1
    return None
    '''
    return calculate_num_features(disabled_var_labels) + 1

def calculate_target_rate(data):
    if data is None:
        return None
    return data[targetLabel].mean()


def calculate_variable_property_vars(data,disabled_var_labels):
    return pd.DataFrame({
        "property": ["num_obs", "num_features", "num_parameters", "target_rate"],
        "value": [calculate_num_obs(data),
                  calculate_num_features(disabled_var_labels),
                  calculate_num_parameters(disabled_var_labels),
                  calculate_target_rate(data)]})



def calculate_train_data_profile(data):
    var_list = data.columns.tolist()
    numeric_list = data.select_dtypes(include=settings.OPTIONS["numeric_types"]).columns.tolist()
    df_datetime = df_utils.get_all_datetime_format(data)
    datetime_list = df_datetime[df_datetime["is_datetime_format"] == True]["variable"].tolist()
    df_missing = df_utils.get_percent_missing(data)

    ret = {
        "variable": [],
        "type": [],
        "percent_missing": [],
        "num_unique_values": [],
        "is_probability": [],
        "is_target": [],
        "is_binned_feature": [],
    }
    for var in var_list:
        ret["variable"].append(var)
        if var in datetime_list:
            ret["type"].append("datetime")
        elif var in numeric_list:
            ret["type"].append("numeric")
        else:
            ret["type"].append("categorical")
        ret["percent_missing"].append(df_missing.loc[var, "perc_missing"])
        ret["num_unique_values"].append(len(data[var].unique()))
        if var == probabilityLabel:
            ret["is_probability"].append(True)
        else:
            ret["is_probability"].append(False)
        if var == targetLabel:
            ret["is_target"].append(True)
        else:
            ret["is_target"].append(False)
        if var in df_binned:
            ret["is_binned_feature"].append(True)
        else:
            ret["is_binned_feature"].append(False)

    return pd.DataFrame.from_dict(ret)




disabledVariables=calculate_disabled_var_labels_from_dataset(analyzeOkPressedData)
panelVars=calculate_variable_selection_panel_vars(analyzeOkPressedData)
varProperty=calculate_variable_property_vars(analyzeOkPressedData,disabledVariables)

dataProfilers=calculate_train_data_profile(analyzeOkPressedData)
x=10
