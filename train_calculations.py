from artifacts import Artifacts
from binning_settings import BinAsMode
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


class TrainCalculate:
    def __init__(self, dataset: Dataset, settings: TrainSettings,
                 artifacts: Artifacts):
        self.ds = dataset
        self.settings = settings
        self.bin_vars_df = None
        if artifacts.binning_artifacts.bin_vars_df is not None:
            self.bin_vars_df = artifacts.binning_artifacts.bin_vars_df.copy()

        self.bin_as_vars_df = None
        # ToDo: Consider migrating next part to binning artifacts
        if artifacts.binning_artifacts.data_vars_df is not None and self.bin_vars_df is not None:
            self.bin_as_vars_df = \
                artifacts.binning_artifacts.data_vars_df[["variable", "bin_as"]].copy()
            temp1 = self.bin_vars_df[["variable_parent", "variable"]].copy()
            self.bin_as_vars_df = \
                self.bin_as_vars_df.merge(temp1,
                                          left_on=["variable"],
                                          right_on=["variable_parent"])
            self.bin_as_vars_df["variable"] = self.bin_as_vars_df["variable_y"]
            self.bin_as_vars_df = self.bin_as_vars_df[["variable_parent", "variable", "bin_as"]]

        self.bin_vars_list = None
        if self.bin_vars_df is not None:
            self.bin_vars_list = list(self.bin_vars_df["variable"].unique())
        self.bin_vars_dict = None
        if artifacts.analyze_artifacts.correlation_analysis_df is not None:
            temp = artifacts.analyze_artifacts.correlation_analysis_df[["variable", "iv"]].copy()
            temp.set_index("variable", inplace=True)
            self.bin_vars_dict = temp.to_dict()["iv"]
        elif artifacts.binning_artifacts.binned_vars_dict is not None:
            self.bin_vars_dict = artifacts.binning_artifacts.binned_vars_dict.copy()
        self.woe = WeightOfEvidence()

        if not self.settings:
            self.settings = TrainSettings()

    def calculate_num_obs(self):
        if self.ds is None:
            return None
        return self.ds.data.shape[0]

    def calculate_num_features(self, disabled_var_labels):
        return len(self.ds.model_qualified_var_labels) - len(disabled_var_labels)

    def calculate_num_parameters(self, disabled_var_labels):
        if self.settings.model_settings.model_framework == ModelType.LOGISTIC_REGRESSION:
            return self.calculate_num_features(disabled_var_labels) + 1
        return None

    def calculate_target_rate(self):
        if self.ds is None:
            return None
        return self.ds.data[self.ds.target_label].mean()

    def calculate_disabled_var_labels_from_dataset(self):
        all_vars = list(self.ds.data.columns)
        var_names = self.ds.model_qualified_var_labels
        disabled_var_labels = []
        for i in self.ds.excluded_vars:
            var = all_vars[i]
            if var in var_names:
                disabled_var_labels.append(var)
        return disabled_var_labels

    def calculate_disabled_var_labels_panel(self):
        return self.calculate_disabled_var_labels_from_dataset()

    def calculate_variable_selection_panel_vars(self):
        disabled_var_labels = self.calculate_disabled_var_labels_panel()
        var_list = self.ds.model_qualified_var_labels
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

    def calculate_variable_property_vars(self, disabled_var_labels):
        return pd.DataFrame({
            "property": ["num_obs", "num_features", "num_parameters", "target_rate"],
            "value": [self.calculate_num_obs(),
                      self.calculate_num_features(disabled_var_labels),
                      self.calculate_num_parameters(disabled_var_labels),
                      self.calculate_target_rate()]})

    def calculate_train_data_profile(self):
        var_list = self.ds.data.columns.tolist()
        numeric_list = self.ds.data.select_dtypes(include=settings.OPTIONS["numeric_types"]).columns.tolist()
        df_datetime = df_utils.get_all_datetime_format(self.ds.data)
        datetime_list = df_datetime[df_datetime["is_datetime_format"] == True]["variable"].tolist()
        df_missing = df_utils.get_percent_missing(self.ds.data)

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
            ret["num_unique_values"].append(len(self.ds.data[var].unique()))
            if var == self.ds.probability_label:
                ret["is_probability"].append(True)
            else:
                ret["is_probability"].append(False)
            if var == self.ds.target_label:
                ret["is_target"].append(True)
            else:
                ret["is_target"].append(False)
            if var in self.ds.binned_feature_var_labels:
                ret["is_binned_feature"].append(True)
            else:
                ret["is_binned_feature"].append(False)

        return pd.DataFrame.from_dict(ret)

    def update_settings(self, settings: TrainSettings):
        self.settings = settings

    def train_model(self, model_type, disabled_var_labels: list[str], model_parameters):
        var_list = [x for x in self.ds.model_qualified_var_labels if x not in disabled_var_labels]
        data_x = self.ds.data[var_list].copy()
        target_y = self.ds.data[self.ds.target_label].copy()

        if model_type == ModelType.LOGISTIC_REGRESSION:
            ret = LogisticRegression(data_x=data_x, target_y=target_y, **model_parameters)
            ret.train()
            return ret
        elif model_type == ModelType.RANDOM_FOREST:
            ret = RandomForest(data_x=data_x, target_y=target_y, **model_parameters)
            ret.train()
            return ret

    def calculate_calibration_vars(self, model: Model):
        if model is None:
            return None

        target_rate = self.calculate_target_rate()
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

    @staticmethod
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

    def calculate_iv(self, modelled_var_list: list[str]):
        var_list = [self.ds.target_label] + modelled_var_list
        df = self.ds.data[var_list].copy()
        df_iv = pd.DataFrame()
        # Extract the variables whose IVs have been calculated before and their corresponding IV
        bin_already_list = []
        if self.bin_vars_dict is not None:
            bin_already_list = \
                [x for x in modelled_var_list if x in list(self.bin_vars_dict.keys())]

        iv_already_list: list[float] = []
        for var in bin_already_list:
            iv_already_list.append(self.bin_vars_dict[var])

        # Extract the variables whose IVs have not been calculated before and calculate their IV
        bin_first_time_list = [var for var in modelled_var_list if var not in bin_already_list]
        iv_first_time_list = []
        if len(bin_first_time_list) > 0:
            df_iv_first_time = \
                self.woe.calculate_woe_iv_from_df(df=df,
                                                  var_list=bin_first_time_list,
                                                  target_label=self.ds.target_label)
            df_iv_first_time = df_iv_first_time[["variable", "iv"]].drop_duplicates("variable")
            iv_first_time_list = list(df_iv_first_time["iv"])

        df_iv["variable"] = bin_already_list + bin_first_time_list
        df_iv["iv"] = iv_already_list + iv_first_time_list

        df_iv.sort_values(by=["iv", "variable"], ascending=[False, True], inplace=True)
        df_iv.reset_index(drop=True, inplace=True)
        return df_iv

    def calculate_vif(self, modelled_var_list: list[str]):
        vif_list = [np.nan]  # Intercept set to NULL
        warnings.filterwarnings("ignore")  # ToDo: Consider removing / replacing (for infinity)
        if len(modelled_var_list) == 1:
            vif_list.append(np.nan)
        elif len(modelled_var_list) > 1:
            vif_list.extend(
                [variance_inflation_factor(self.ds.data[modelled_var_list].values, i) for
                 i in range(len(modelled_var_list))])
        warnings.filterwarnings("default")  # ToDo: Consider removing / replacing
        return pd.DataFrame({"variable": ["intercept"] + modelled_var_list, "vif": vif_list})

    def calculate_binned_non_num_list(self, modelled_var_list: list[str]):
        if self.bin_as_vars_df is None:
            return []
        non_num_bin_as_vars_df = \
            self.bin_as_vars_df[
                self.bin_as_vars_df["bin_as"].isin(
                    BinAsMode.non_numeric_modes())].reset_index(drop=True).copy()
        if non_num_bin_as_vars_df.shape[0] == 0:
            return []
        return [x for x in modelled_var_list if x in list(non_num_bin_as_vars_df["variable"])]

    def calculate_direction_with_target(self, model: Model):
        if model is None:
            return None
        modelled_var_list = model.variable_list
        binned_non_num_list = self.calculate_binned_non_num_list(modelled_var_list)
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

    def calculate_binned_num_list(self, modelled_var_list: list[str]):
        if self.bin_as_vars_df is None:
            return []
        num_bin_as_vars_df = \
            self.bin_as_vars_df[
                self.bin_as_vars_df["bin_as"].isin(
                    BinAsMode.numeric_modes())].reset_index(drop=True).copy()
        if num_bin_as_vars_df.shape[0] == 0:
            return []
        return [x for x in modelled_var_list if x in list(num_bin_as_vars_df["variable"])]

    def calculate_non_binned_num_list(self, modelled_var_list: list[str]):
        binned_non_num_list = self.calculate_binned_non_num_list(modelled_var_list)
        binned_num_list = self.calculate_binned_non_num_list(modelled_var_list)
        return [x for x in modelled_var_list if x not in binned_non_num_list + binned_num_list]

    def calculate_parent_var_label(self, binned_var_label: list[str]):
        if self.bin_as_vars_df is None:
            return None
        if binned_var_label not in list(self.bin_as_vars_df["variable"]):
            return None
        return list(self.bin_as_vars_df[
                        self.bin_as_vars_df["variable"] == binned_var_label]["variable_parent"])[0]

    @staticmethod
    def shock_series(series, cap: bool = False):
        if cap:
            series_max = series.max()
            return pd.Series(np.select([series + series.std() > series_max],
                                       [series_max], default=series + series.std()))
        return series + series.std()

    def calculate_sensitivity_binned(self, model: Model, binned_var_label: list[str]):
        parent_var_label = self.calculate_parent_var_label(binned_var_label)
        if parent_var_label is None:
            return np.nan
        modelled_var_list = model.variable_list
        data_x_shocked = self.ds.data[modelled_var_list].copy()
        avg_prob = model.get_average_probability(data_x_shocked)
        data_x_shocked[binned_var_label] = self.ds.data[parent_var_label].copy()
        data_x_shocked[binned_var_label] = \
            self.shock_series(data_x_shocked[binned_var_label], cap=True)
        data_x_shocked[binned_var_label] = self.woe.map_to_woe(data_x_shocked[binned_var_label],
                                                               self.bin_vars_df[
                                                                   self.bin_vars_df["variable"] ==
                                                                   binned_var_label])
        avg_prob_shocked = model.get_average_probability(data_x_shocked)
        return (avg_prob_shocked - avg_prob) / avg_prob

    def calculate_sensitivity_non_binned_num(self, model: Model, var_label: list[str]):
        modelled_var_list = model.variable_list
        data_x_shocked = self.ds.data[modelled_var_list].copy()
        avg_prob = model.get_average_probability(data_x_shocked)
        data_x_shocked[var_label] = self.shock_series(data_x_shocked[var_label], cap=False)
        avg_prob_shocked = model.get_average_probability(data_x_shocked)
        return (avg_prob_shocked - avg_prob) / avg_prob

    def calculate_sensitivity(self, model: Model):
        if model is None:
            return None
        modelled_var_list = model.variable_list
        var_list = ["intercept"] + modelled_var_list
        binned_num_list = self.calculate_binned_num_list(modelled_var_list)
        non_binned_num_list = self.calculate_non_binned_num_list(modelled_var_list)

        sensitivity_list = []
        for var in var_list:
            if var in binned_num_list:
                sensitivity_list.append(self.calculate_sensitivity_binned(model, var))
            elif var in non_binned_num_list:
                sensitivity_list.append(self.calculate_sensitivity_non_binned_num(model, var))
            else:
                sensitivity_list.append(np.nan)

        df_sensitivity = pd.DataFrame({"variable": var_list, "sensitivity": sensitivity_list})
        return df_sensitivity

    def calculate_model_variable_dynamics_vars(self, model: Model):
        if model is None:
            return None

        if model.model_type == ModelType.LOGISTIC_REGRESSION:
            modelled_var_list = model.variable_list
            model_coef_df = model.coefficients
            p_values_df = model.p_values
            iv_df = self.calculate_iv(modelled_var_list)
            vif_df = self.calculate_vif(modelled_var_list)
            direction_df = self.calculate_direction_with_target(model)
            sensitivity_df = self.calculate_sensitivity(model)
            df_output = model_coef_df
            df_output = df_output.merge(p_values_df, how="left", on="variable")
            df_output = df_output.merge(iv_df, how="left", on="variable")
            df_output = df_output.merge(vif_df, how="left", on="variable")
            df_output = df_output.merge(direction_df, how="left", on="variable")
            df_output = df_output.merge(sensitivity_df, how="left", on="variable")
        elif model.model_type == ModelType.RANDOM_FOREST:
            df_output = model.feature_importances
        return df_output

    @staticmethod
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

    @staticmethod
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
