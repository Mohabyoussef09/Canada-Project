import numpy as np
import pandas as pd

from train_model import ModelType
import display_utils


class TrainDisplay:
    def __init__(self):
        self.variable_selection_display_column_list = \
            ["variable", "include_exclude", "manually_set"]
        self.variable_selection_mapping = {
            "variable": "Variable",
            "include_exclude": "Include/Exclude",
            "manually_set": "Manual/Panel",
        }

        self.variable_property_mapping = {
            "value": "",
        }

        self.variable_property_labels_mapping = {
            "num_obs": "# observations",
            "num_features": "# features",
            "num_parameters": "# parameters",
            "target_rate": "Target rate"}

        self.calibration_mapping = {
            "value": "",
        }
        self.calibration_labels_mapping = {
            "calibration_setting": "Calib. setting",
            "target_rate": "Target rate",
            "average_probability": "Avg. probability",
            "difference": "Difference",
            "percentage_change": "% change",
        }

        self.kpis_mapping = {
            "value": "",
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

    def display_variable_selection_vars(self, variable_selection_vars: pd.DataFrame):
        ret = variable_selection_vars[self.variable_selection_display_column_list].copy()
        ret["include_exclude"] = [str(x) for x in list(ret["include_exclude"])]
        ret["manually_set"] = \
            np.select(
                [ret["manually_set"].isin([True]), ret["manually_set"].isin([False])],
                ["Manual", "Panel"])
        ret = ret[self.variable_selection_display_column_list]
        ret.rename(columns=self.variable_selection_mapping, inplace=True)
        return ret

    def display_variable_property_vars(self, variable_property_vars: pd.DataFrame):
        ret = variable_property_vars.copy()
        value_list = []
        for row in range(ret.shape[0]):
            if ret.at[row, "value"] is None or np.isnan(ret.at[row, "value"]):
                value_list.append("")
            else:
                if ret.at[row, "property"] == "target_rate":
                    value_list.append("{:,.2%}".format(ret.at[row, "value"]))
                else:
                    value_list.append("{:,.0f}".format(ret.at[row, "value"]))
            ret.at[row, "property"] = self.variable_property_labels_mapping[ret.at[row, "property"]]

        ret["value"] = value_list
        ret.rename(columns=self.variable_property_mapping, inplace=True)
        ret.set_index("property", inplace=True)
        return ret

    def display_calibration_vars(self, calibration_vars: pd.DataFrame):
        ret = calibration_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "calibration_criterion"] = \
                self.calibration_labels_mapping[ret.at[row, "calibration_criterion"]]
        ret["value"] = display_utils.fmt_apply(ret, "value", 2, "%")
        ret.rename(columns=self.calibration_mapping, inplace=True)
        ret.set_index("calibration_criterion", inplace=True)
        return ret

    def display_kpis_vars(self, kpis_vars: pd.DataFrame):
        ret = kpis_vars.copy()
        value_list = []
        for row in range(ret.shape[0]):
            if ret.at[row, "kpi"] == "aic":
                value_list.append("{:,.0f}".format(ret.at[row, "value"]))
            else:
                value_list.append("{:,.0%}".format(ret.at[row, "value"]))
            ret.at[row, "kpi"] = self.kpis_labels_mapping[ret.at[row, "kpi"]]
        ret["value"] = value_list
        ret.rename(columns=self.kpis_mapping, inplace=True)
        ret.set_index("kpi", inplace=True)
        return ret

    def display_model_variable_dynamics_vars(self, model_variable_dynamics_var: pd.DataFrame,
                                             model_type: ModelType) -> pd.DataFrame:
        ret = model_variable_dynamics_var.copy()

        if model_type == ModelType.LOGISTIC_REGRESSION:
            ret["temp_index"] = [0] + [1] * (ret.shape[0] - 1)
            ret["abs_sensitivity"] = abs(ret["sensitivity"])
            ret.sort_values(by=["temp_index", "iv", "abs_sensitivity"],
                            ascending=[True, False, False],
                            inplace=True)
            ret.drop(columns=["temp_index", "abs_sensitivity"], inplace=True)
            ret["coefficient"] = display_utils.fmt_apply(ret, "coefficient", 2, "f")
            ret["p_value"] = np.select([ret["p_value"] < 0.0001], ["<0.01%"],
                                       default=display_utils.fmt_apply(ret, "p_value", 2, "%"))
            ret["iv"] = display_utils.fmt_apply(ret, "iv", 0, "%")
            ret["iv"].replace("nan%", "N/A", inplace=True)
            ret["vif"] = display_utils.fmt_apply(ret, "vif", 1, "f")
            ret["vif"].replace("nan", "N/A", inplace=True)
            ret["vif"].replace("inf", "∞", inplace=True)
            ret["direction"] = \
                np.select([ret["direction"] == "positive",
                           ret["direction"] == "negative",
                           ~ret["direction"].isin(["positive", "negative"])],
                          ["Positive", "Negative", "None"])
            ret["sensitivity"] = display_utils.fmt_apply(ret, "sensitivity", 1, "%")
            ret["sensitivity"].replace("nan%", "N/A", inplace=True)

            ret.rename(columns=self.model_variable_dynamics_mapping_logistic_regression,
                       inplace=True)

        elif model_type == ModelType.RANDOM_FOREST:
            ret["importance"] = display_utils.fmt_apply(ret, "importance", 1, "%")
            ret.rename(columns=self.model_variable_dynamics_mapping_random_forest,
                       inplace=True)

        return ret

    @staticmethod
    def display_ranking(df_rank: pd.DataFrame):
        lower_limit_list_main = \
            list(df_rank.apply(lambda data_x: "{:,.2%}".format(data_x["lower_limit"]), axis=1))
        upper_limit_list_main = \
            list(df_rank.apply(lambda data_x: "{:,.2%}".format(data_x["upper_limit"]), axis=1))
        bin_list = []
        target_rate_list = list(df_rank["target_rate"])
        n = len(lower_limit_list_main)
        for iteration in range(n):
            right_interval = ")"
            if iteration == n - 1:
                right_interval = "]"
            bin_list.append("[" + lower_limit_list_main[iteration] + ", " +
                            upper_limit_list_main[iteration] + right_interval)
        return bin_list, target_rate_list

    @staticmethod
    def display_distribution(df_distribution: pd.DataFrame):
        lower_limit_list_main = \
            list(df_distribution.apply(lambda data_x: "{:,.2%}".format(data_x["lower_limit"]),
                                       axis=1))
        bin_list = []
        concentration_list = list(df_distribution["concentration"])
        n = len(lower_limit_list_main)
        for iteration in range(n):
            bin_list.append(lower_limit_list_main[iteration])

        # Final label for display purposes
        bin_list.append("100.00%")
        concentration_list.append(0.0)
        return bin_list, concentration_list
