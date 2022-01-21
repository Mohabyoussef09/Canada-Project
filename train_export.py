from train_settings import TrainSettings
import numpy as np
import pandas as pd

from train_model import ModelType


class TrainExport:
    def __init__(self, settings: TrainSettings):
        self.settings = settings

        self.model_settings_mapping = {
            "model_setting": "Model setting",
            "value": "Value",
        }
        self.model_settings_labels_mapping_logistic_regression = {
            "model_framework": "Model framework",
        }
        self.model_settings_labels_mapping_random_forest = {
            "model_framework": "Model framework",
            "num_trees": "# trees",
            "max_depth": "Max. depth",
            "min_samples_split": "Min. samples split",
            "min_samples_leaf": "Min. samples leaf",
        }

        self.variable_selection_mapping = {
            "variable": "Variable",
            "include_exclude_panel": "Include/Exclude (default settings)",
            "include_exclude": "Include/Exclude (actual settings)",
            "manually_set": "Manual/Panel",
        }

        self.variable_property_mapping = {
            "property": "Property",
            "value": "Value",
        }
        self.variable_property_labels_mapping = {
            "num_obs": "# observations",
            "num_features": "# features",
            "num_parameters": "# parameters",
            "target_rate": "Target rate"}

        self.variable_selection_panel_labels_mapping = {
            "select_all": "Select all",
            "binned_only": "Binned only",
        }

        self.calibration_mapping = {
            "calibration_criterion": "Calibration criterion",
            "value": "Value",
        }
        self.calibration_labels_mapping = {
            "calibration_setting": "Calibration setting",
            "target_rate": "Target rate",
            "average_probability": "Average probability",
            "difference": "Difference",
            "percentage_change": "Percentage change",
        }
        self.kpis_mapping = {
            "kpi": "KPI",
            "value": "Value",
        }
        self.kpis_labels_mapping = {
            "auroc": "AUROC",
            "ks": "KS",
            "aic": "AIC",
        }
        self.model_variable_dynamics_mapping_logistic_regression = {
            "variable": "Variable",
            "coefficient": "Coefficient",
            "p_value": "p-value",
            "iv": "IV",
            "vif": "VIF",
            "direction": "Direction with target",
            "sensitivity": "Sensitivity (+1 std)",
        }
        self.model_variable_dynamics_mapping_random_forest = {
            "variable": "Variable",
            "importance": "Importance",
        }
        self.ranking_mapping = {
            "bin_num": "Bin #",
            "lower_limit": "Lower limit",
            "upper_limit": "Upper limit",
            "bin": "Bin",
            "num_obs": "# obs",
            "num_events": "# events",
            "target_rate": "Target rate",
        }
        self.distribution_mapping = {
            "bin_num": "Bin #",
            "lower_limit": "Lower limit",
            "upper_limit": "Upper limit",
            "num_obs": "# obs",
            "concentration": "% concentration",
        }

    def export_model_settings(self, model_type):
        ret = self.settings.model_settings_df
        for row in range(ret.shape[0]):
            if model_type == ModelType.LOGISTIC_REGRESSION:
                ret.at[row, "model_setting"] = \
                    self.model_settings_labels_mapping_logistic_regression[ret.at[row,
                                                                                  "model_setting"]]
            elif model_type == ModelType.RANDOM_FOREST:
                ret.at[row, "model_setting"] = \
                    self.model_settings_labels_mapping_random_forest[ret.at[row, "model_setting"]]
        ret.rename(columns=self.model_settings_mapping, inplace=True)
        return ret

    def export_variable_selection_vars(self, variable_selection_vars: pd.DataFrame):
        ret = variable_selection_vars.copy()
        ret["include_exclude_panel"] = [str(x) for x in list(ret["include_exclude_panel"])]
        ret["include_exclude"] = [str(x) for x in list(ret["include_exclude"])]
        ret["manually_set"] = \
            np.select(
                [ret["manually_set"].isin([True]), ret["manually_set"].isin([False])],
                ["Manual", "Panel"])
        ret.rename(columns=self.variable_selection_mapping, inplace=True)
        return ret

    def export_variable_property_vars(self, variable_property_vars: pd.DataFrame):
        ret = variable_property_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "property"] = self.variable_property_labels_mapping[ret.at[row, "property"]]
        ret.rename(columns=self.variable_property_mapping, inplace=True)
        return ret

    def export_calibration_vars(self, calibration_vars: pd.DataFrame):
        ret = calibration_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "calibration_criterion"] = \
                self.calibration_labels_mapping[ret.at[row, "calibration_criterion"]]
        ret.rename(columns=self.calibration_mapping, inplace=True)
        return ret

    def export_kpis_vars(self, kpis_vars: pd.DataFrame):
        ret = kpis_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "kpi"] = self.kpis_labels_mapping[ret.at[row, "kpi"]]
        ret.rename(columns=self.kpis_mapping, inplace=True)
        return ret

    def export_model_variable_dynamics_vars(self, model_variable_dynamics_vars: pd.DataFrame,
                                            model_type: ModelType):
        ret = model_variable_dynamics_vars.copy()

        if model_type == ModelType.LOGISTIC_REGRESSION:
            ret["iv"] = ret["iv"].astype(str)
            ret["iv"].replace("nan", "N/A", inplace=True)
            ret["vif"] = ret["vif"].astype(str)
            ret["vif"].replace("nan", "N/A", inplace=True)
            ret["vif"].replace("inf", "Infinity", inplace=True)
            ret["direction"] = \
                np.select([ret["direction"] == "positive",
                           ret["direction"] == "negative",
                           ~ret["direction"].isin(["positive", "negative"])],
                          ["Positive", "Negative", "None"])
            ret["sensitivity"] = ret["sensitivity"].astype(str)
            ret["sensitivity"].replace("nan", "N/A", inplace=True)
            ret.rename(columns=self.model_variable_dynamics_mapping_logistic_regression,
                       inplace=True)
        elif model_type == ModelType.RANDOM_FOREST:
            ret.rename(columns=self.model_variable_dynamics_mapping_random_forest,
                       inplace=True)
        return ret

    def export_ranking_vars(self, ranking_vars):
        ret = ranking_vars.copy()
        ret.rename(columns=self.ranking_mapping, inplace=True)
        return ret

    def export_distribution_vars(self, distribution_vars):
        ret = distribution_vars.copy()
        ret.rename(columns=self.distribution_mapping, inplace=True)
        return ret

    def export_all(self,
                   model_type: ModelType,
                   variable_selection_vars: pd.DataFrame,
                   variable_property_vars: pd.DataFrame,
                   calibration_vars: pd.DataFrame,
                   kpis_vars: pd.DataFrame,
                   model_variable_dynamics_vars: pd.DataFrame,
                   ranking_vars: pd.DataFrame,
                   distribution_vars: pd.DataFrame):
        df_filler = pd.DataFrame()
        df_filler[""] = [""] * 1
        df1 = self.export_model_settings(model_type)
        df2 = self.export_variable_property_vars(variable_property_vars)
        df3 = self.export_variable_selection_vars(variable_selection_vars)
        df4 = self.export_kpis_vars(kpis_vars)
        df5 = self.export_calibration_vars(calibration_vars)
        df6 = self.export_model_variable_dynamics_vars(model_variable_dynamics_vars, model_type)
        df7 = self.export_ranking_vars(ranking_vars)
        df8 = self.export_distribution_vars(distribution_vars)
        ret = pd.concat(
            [df1, df_filler,
             df2, df_filler,
             df3, df_filler,
             df4, df_filler,
             df5, df_filler,
             df6, df_filler,
             df7, df_filler,
             df8],
            axis=1,
        )
        return ret
