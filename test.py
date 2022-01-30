import df_utils
from binning_calculations import calculate_data_variables,calculate_bin_variables,apply_binning
from analyze_calculations import calculate_distribution_variables,calculate_volatility_variables,calculate_correlation_variables
from dataset import Dataset
from main_window import MainWindow
from train_new import *
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

def update_excluded_vars_list(data,invalid_vars):
    excluded_vars = invalid_vars.copy()

    if ( min(excluded_vars) < 0  or max(excluded_vars) >= data.shape[1]):
        return sorted(list(set(invalid_vars).union(set(excluded_vars))))
    return data


excludedVar=invalid_vars
user_excluded_var_indices = [i for i, x in enumerate(list(data.columns)) if  i in invalid_vars]
analyzeOkPressedData=update_excluded_vars_list(data,user_excluded_var_indices)


############################################
#train screen
###########################################

modelType="Logistic regression"

disabledVariableLabels=calculate_disabled_var_labels_from_dataset(analyzeOkPressedData,modelQualifiedVarLabels,invalid_vars)
panelVars=calculate_variable_selection_panel_vars(analyzeOkPressedData,modelQualifiedVarLabels,invalid_vars)
varProperty=calculate_variable_property_vars(analyzeOkPressedData,targetLabel,disabledVariableLabels,modelQualifiedVarLabels,modelType)

dataProfilers=calculate_train_data_profile(analyzeOkPressedData,probabilityLabel,targetLabel,bin_vars)
x=10



'''to_str = {
            self.LOGISTIC_REGRESSION: "Logistic regression",
            self.RANDOM_FOREST: "Random forest",
        }
'''
model = train_model(analyzeOkPressedData, modelType,modelQualifiedVarLabels,targetLabel, disabledVariableLabels, dict())

kpisVariables=calculate_kpis_vars(model)
collaborationVariables=calculate_calibration_vars(analyzeOkPressedData,targetLabel,model)
distributionVariables=calculate_distribution_vars(model)
rankingVariables=calculate_ranking_vars(model)
modelVariables=calculate_model_variable_dynamics_vars(analyzeOkPressedData, targetLabel,bin_vars,modelType,model)


trainDataFrame=analyzeOkPressedData.copy()
trainDataFrame[probabilityLabel]=model.prob_y
print(trainDataFrame.sample())



############################
############################


