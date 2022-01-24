from artifacts import Artifacts
from binning_settings import BinAsMode
from constants import LOGISTIC_REGRESSION, RANDOM_FOREST
from train_settings import ModelType, TrainSettings
from dataset import Dataset
from modellicity.src.modellicity.stats.weight_of_evidence import WeightOfEvidence
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from train_model import LogisticRegression, Model, RandomForest
from train_settings import IncludeExclude
from modellicity.src.modellicity.settings import settings
from typing import Any
import numpy as np
import pandas as pd
import warnings
import df_utils


def calculate_num_obs(data):
    if data is None:
        return None
    return data.shape[0]


def calculate_num_features(model_qualified_var_labels, disabled_var_labels):
    return len(model_qualified_var_labels) - len(disabled_var_labels)


def calculate_num_parameters(disabled_var_labels,modelType):
    '''
    if settings.model_settings.model_framework == modelType:
        return calculate_num_features(disabled_var_labels) + 1
    '''
    return None


def calculate_target_rate(data,target_label):
    if data is None:
        return None
    return data[target_label].mean()


def calculate_disabled_var_labels_from_dataset(data,model_qualified_var_labels,excluded_vars):
    all_vars = list(data.columns)
    var_names = model_qualified_var_labels
    disabled_var_labels = []
    for i in excluded_vars:
        var = all_vars[i]
        if var in var_names:
            disabled_var_labels.append(var)
    return disabled_var_labels


def calculate_disabled_var_labels_panel(data,model_qualified_var_labels,invalid_vars):
    return calculate_disabled_var_labels_from_dataset(data,model_qualified_var_labels,invalid_vars)


def calculate_variable_selection_panel_vars(data,model_qualified_var_labels,invalid_vars):
    disabled_var_labels = calculate_disabled_var_labels_panel(data,model_qualified_var_labels,invalid_vars)
    var_list = model_qualified_var_labels
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


def calculate_variable_property_vars(data,target_label,disabled_var_labels,modelQualifiedVarLabels,modelType):
    return pd.DataFrame({
        "property": ["num_obs", "num_features", "num_parameters", "target_rate"],
        "value": [calculate_num_obs(data),
                  calculate_num_features(disabled_var_labels,modelQualifiedVarLabels),
                  calculate_num_parameters(disabled_var_labels,modelType),
                  calculate_target_rate(data,target_label)]})


def calculate_train_data_profile(data,probability_label,target_label,binned_feature_var_labels):
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
        if var == probability_label:
            ret["is_probability"].append(True)
        else:
            ret["is_probability"].append(False)
        if var == target_label:
            ret["is_target"].append(True)
        else:
            ret["is_target"].append(False)
        if var in binned_feature_var_labels:
            ret["is_binned_feature"].append(True)
        else:
            ret["is_binned_feature"].append(False)

    return pd.DataFrame.from_dict(ret)


def train_model(data, model_type,model_qualified_var_labels,target_label, disabled_var_labels: list[str], model_parameters):
    var_list = [x for x in model_qualified_var_labels if x not in disabled_var_labels]
    data_x =data[var_list].copy()
    target_y = data[target_label].copy()

    if model_type == LOGISTIC_REGRESSION:
        ret = LogisticRegression(data_x=data_x, target_y=target_y, **model_parameters)
        ret.train()
        return ret
    elif model_type == RANDOM_FOREST:
        ret = RandomForest(data_x=data_x, target_y=target_y, **model_parameters)
        ret.train()
        return ret


def calculate_calibration_vars(data,targetLabel,model: Model):
    if model is None:
        return None

    target_rate = calculate_target_rate(data,targetLabel)
    average_probability = model.prob_y.mean()
    difference = average_probability - target_rate
    percentage_change = difference / target_rate
    return pd.DataFrame({
        "calibration_criterion": [
            "target_rate", "average_probability", "difference",
            "percentage_change"],
        "value": [target_rate,
                  average_probability,
                  difference,
                  percentage_change]})


def calculate_kpis_vars(model: Model):
    if model is None:
        return None
    tmp = pd.DataFrame()
    tmp["target"], tmp["prob"] = model.target_y, model.prob_y
    tmp_event = tmp[tmp["target"] == 1]
    tmp_non_event = tmp[tmp["target"] == 0]
    auroc = roc_auc_score(model.target_y, model.prob_y)
    ks = stats.ks_2samp(tmp_event["prob"], tmp_non_event["prob"])[0]
    return pd.DataFrame({"kpi": ["auroc", "ks"], "value": [auroc, ks]})


def calculate_iv(data, target_label,bin_vars_dict,modelled_var_list: list[str]):
    var_list = [target_label] + modelled_var_list
    woe = WeightOfEvidence()
    df = data[var_list].copy()
    df_iv = pd.DataFrame()
    # Extract the variables whose IVs have been calculated before and their corresponding IV
    bin_already_list = []
    if bin_vars_dict is not None:
        bin_already_list = \
            [x for x in modelled_var_list if x in list(bin_vars_dict.keys())]

    iv_already_list: list[float] = []
    for var in bin_already_list:
        iv_already_list.append(bin_vars_dict[var])

    # Extract the variables whose IVs have not been calculated before and calculate their IV
    bin_first_time_list = [var for var in modelled_var_list if var not in bin_already_list]
    iv_first_time_list = []
    if len(bin_first_time_list) > 0:
        df_iv_first_time = \
            woe.calculate_woe_iv_from_df(df=df,
                                              var_list=bin_first_time_list,
                                              target_label=target_label)
        df_iv_first_time = df_iv_first_time[["variable", "iv"]].drop_duplicates("variable")
        iv_first_time_list = list(df_iv_first_time["iv"])

    df_iv["variable"] = bin_already_list + bin_first_time_list
    df_iv["iv"] = iv_already_list + iv_first_time_list

    df_iv.sort_values(by=["iv", "variable"], ascending=[False, True], inplace=True)
    df_iv.reset_index(drop=True, inplace=True)
    return df_iv


def calculate_vif(data,modelled_var_list: list[str]):
    vif_list = [np.nan]  # Intercept set to NULL
    warnings.filterwarnings("ignore")  # ToDo: Consider removing / replacing (for infinity)
    if len(modelled_var_list) == 1:
        vif_list.append(np.nan)
    elif len(modelled_var_list) > 1:
        vif_list.extend(
            [variance_inflation_factor(data[modelled_var_list].values, i) for
             i in range(len(modelled_var_list))])
    warnings.filterwarnings("default")  # ToDo: Consider removing / replacing
    return pd.DataFrame({"variable": ["intercept"] + modelled_var_list, "vif": vif_list})



def calculate_binned_non_num_list(bin_as_vars_df,modelled_var_list: list[str]):
    if bin_as_vars_df is None:
        return []
    non_num_bin_as_vars_df = \
        bin_as_vars_df[
            bin_as_vars_df["bin_as"].isin(
                BinAsMode.non_numeric_modes())].reset_index(drop=True).copy()
    if non_num_bin_as_vars_df.shape[0] == 0:
        return []
    return [x for x in modelled_var_list if x in list(non_num_bin_as_vars_df["variable"])]



def calculate_direction_with_target(bin_as_vars_df,model: Model):
    if model is None:
        return None
    modelled_var_list = model.variable_list
    binned_non_num_list = calculate_binned_non_num_list(bin_as_vars_df,modelled_var_list)
    coef_values = np.array(model.coefficients["coefficient"])
    direction_output = np.select([coef_values > 0, coef_values < 0, coef_values == 0],
                                 ["positive", "negative", "none"])
    direction_output[0] = "none"  # Intercept is set to "none"
    df_direction = \
        pd.DataFrame({"variable": ["intercept"] + modelled_var_list,
                      "direction": direction_output})
    # Exclude variables with non-numeric parents
    df_direction["direction"] = \
        np.select([df_direction["variable"].isin(binned_non_num_list)], ["none"],
                  default=df_direction["direction"])
    return df_direction


def calculate_binned_num_list(bin_as_vars_df,modelled_var_list: list[str]):
    if bin_as_vars_df is None:
        return []
    num_bin_as_vars_df = \
        bin_as_vars_df[
            bin_as_vars_df["bin_as"].isin(
                BinAsMode.numeric_modes())].reset_index(drop=True).copy()
    if num_bin_as_vars_df.shape[0] == 0:
        return []
    return [x for x in modelled_var_list if x in list(num_bin_as_vars_df["variable"])]


def calculate_non_binned_num_list(modelled_var_list: list[str]):
    binned_non_num_list = calculate_binned_non_num_list(modelled_var_list)
    binned_num_list = calculate_binned_non_num_list(modelled_var_list)
    return [x for x in modelled_var_list if x not in binned_non_num_list + binned_num_list]


def calculate_parent_var_label(bin_as_vars_df,binned_var_label: list[str]):
    if bin_as_vars_df is None:
        return None
    if binned_var_label not in list(bin_as_vars_df["variable"]):
        return None
    return list(bin_as_vars_df[
                    bin_as_vars_df["variable"] == binned_var_label]["variable_parent"])[0]


def shock_series(series, cap: bool = False):
    if cap:
        series_max = series.max()
        return pd.Series(np.select([series + series.std() > series_max],
                                   [series_max], default=series + series.std()))
    return series + series.std()


def calculate_sensitivity_binned(data,bin_vars_df,woe,model: Model, binned_var_label: list[str]):
    parent_var_label = calculate_parent_var_label(binned_var_label)
    if parent_var_label is None:
        return np.nan
    modelled_var_list = model.variable_list
    data_x_shocked = data[modelled_var_list].copy()
    avg_prob = model.get_average_probability(data_x_shocked)
    data_x_shocked[binned_var_label] = data[parent_var_label].copy()
    data_x_shocked[binned_var_label] = \
        shock_series(data_x_shocked[binned_var_label], cap=True)
    data_x_shocked[binned_var_label] = woe.map_to_woe(data_x_shocked[binned_var_label],
                                                           bin_vars_df[
                                                               bin_vars_df["variable"] ==
                                                               binned_var_label])
    avg_prob_shocked = model.get_average_probability(data_x_shocked)
    return (avg_prob_shocked - avg_prob) / avg_prob


def calculate_sensitivity_non_binned_num(data,model: Model, var_label: list[str]):
    modelled_var_list = model.variable_list
    data_x_shocked = data[modelled_var_list].copy()
    avg_prob = model.get_average_probability(data_x_shocked)
    data_x_shocked[var_label] = shock_series(data_x_shocked[var_label], cap=False)
    avg_prob_shocked = model.get_average_probability(data_x_shocked)
    return (avg_prob_shocked - avg_prob) / avg_prob


def calculate_sensitivity(model: Model):
    if model is None:
        return None
    modelled_var_list = model.variable_list
    var_list = ["intercept"] + modelled_var_list
    binned_num_list = calculate_binned_num_list(modelled_var_list)
    non_binned_num_list = calculate_non_binned_num_list(modelled_var_list)

    sensitivity_list = []
    for var in var_list:
        if var in binned_num_list:
            sensitivity_list.append(calculate_sensitivity_binned(model, var))
        elif var in non_binned_num_list:
            sensitivity_list.append(calculate_sensitivity_non_binned_num(model, var))
        else:
            sensitivity_list.append(np.nan)

    df_sensitivity = pd.DataFrame({"variable": var_list, "sensitivity": sensitivity_list})
    return df_sensitivity


def calculate_model_variable_dynamics_vars(data, target_label,bin_vars_dict,modelType,model: Model):
    if model is None:
        return None

    if modelType == LOGISTIC_REGRESSION:
        modelled_var_list = model.variable_list
        model_coef_df = model.coefficients
        p_values_df = model.p_values
        iv_df = calculate_iv(data, target_label,bin_vars_dict,modelled_var_list)
        vif_df = calculate_vif(data,modelled_var_list)
        direction_df = calculate_direction_with_target(bin_vars_dict,model)
        sensitivity_df = calculate_sensitivity(model)
        df_output = model_coef_df
        df_output = df_output.merge(p_values_df, how="left", on="variable")
        df_output = df_output.merge(iv_df, how="left", on="variable")
        df_output = df_output.merge(vif_df, how="left", on="variable")
        df_output = df_output.merge(direction_df, how="left", on="variable")
        df_output = df_output.merge(sensitivity_df, how="left", on="variable")
    elif modelType == RANDOM_FOREST:
        df_output = model.feature_importances
    return df_output



def calculate_ranking_vars(model: Model):
    if model is None:
        return None
    df_temp = pd.DataFrame({"y": model.target_y, "y_prob": model.prob_y})
    df_rank_output: dict[str, Any] = \
        {"bin_num": [],
         "lower_limit": [],
         "upper_limit": [],
         "bin": [],
         "num_obs": [],
         "num_events": [],
         "target_rate": []}
    n = 5
    percentile_values = [i / n * 100 for i in range(n + 1)]
    percentile_list = \
        list(pd.Series(np.percentile(model.prob_y, percentile_values)).unique())
    percentile_list[0] = 0  # Replace the minimum value with 0
    n = len(percentile_list) - 1
    percentile_list[n] = 1  # Replace the maximum value with 1
    lower_limit_list = percentile_list[0:n]
    upper_limit_list = percentile_list[1:n + 1]
    df_rank_output["bin_num"] = list(range(1, n + 1))
    df_rank_output["lower_limit"] = lower_limit_list
    df_rank_output["upper_limit"] = upper_limit_list
    for i in range(n):
        df_temp_var = df_temp[df_temp["y_prob"] >= lower_limit_list[i]]
        if i == n - 1:
            bin_value = "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + "]"
            df_temp_var = df_temp_var[df_temp_var["y_prob"] <= upper_limit_list[i]]
        else:
            bin_value = "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + ")"
            df_temp_var = df_temp_var[df_temp_var["y_prob"] < upper_limit_list[i]]
        num_obs = df_temp_var.shape[0]
        num_events = df_temp_var["y"].sum()
        if num_obs == 0:
            target_rate_value = 0
        else:
            target_rate_value = num_events / num_obs
        df_rank_output["bin"].append(bin_value)
        df_rank_output["num_obs"].append(num_obs)
        df_rank_output["num_events"].append(num_events)
        df_rank_output["target_rate"].append(target_rate_value)
    return pd.DataFrame().from_dict(df_rank_output)


def calculate_distribution_vars(model: Model):
    if model is None:
        return None
    df_temp = \
        pd.DataFrame({"y": model.target_y, "y_prob": model.prob_y})
    df_dis_output: dict[str, Any] = \
        {"bin_num": [],
         "lower_limit": [],
         "upper_limit": [],
         "num_obs": [],
         "concentration": []}
    num_bins_input = 10
    lower_limit_list = [i / num_bins_input for i in range(num_bins_input)]
    upper_limit_list = [i / num_bins_input for i in range(1, num_bins_input + 1)]
    df_dis_output["bin_num"] = list(range(1, num_bins_input + 1))
    df_dis_output["lower_limit"] = lower_limit_list
    df_dis_output["upper_limit"] = upper_limit_list
    for i in range(num_bins_input):
        df_temp_var = df_temp[df_temp["y_prob"] >= lower_limit_list[i]]
        if i == num_bins_input - 1:
            df_temp_var = df_temp_var[df_temp_var["y_prob"] <= upper_limit_list[i]]
        else:
            df_temp_var = df_temp_var[df_temp_var["y_prob"] < upper_limit_list[i]]
        num_obs = df_temp_var.shape[0]
        concentration = num_obs / df_temp.shape[0]
        df_dis_output["num_obs"].append(num_obs)
        df_dis_output["concentration"].append(concentration)
    return pd.DataFrame().from_dict(df_dis_output)
