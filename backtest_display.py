import pandas as pd
import display_utils


class BacktestDisplay:
    def __init__(self):
        self.kpis_mapping = {
            "train": "Training",
            "backtest": "Backtesting",
        }
        self.kpis_labels_mapping = {
            "auroc": "AUROC",
            "ks": "KS",
        }

        self.calibration_mapping = {
            "train": "Training",
            "backtest": "Backtesting",
        }
        self.calibration_labels_mapping = {
            "num_obs": "# observations",
            "calibration_setting": "Calib. setting",
            "target_rate": "Target rate",
            "average_probability": "Avg. probability",
            "difference": "Difference",
            "percentage_change": "% change",
        }

        self.stability_mapping = {
            "variable": "Variable",
            "psi_var": "PSI",
        }

    def display_kpis_vars(self, kpis_vars: pd.DataFrame):
        ret = kpis_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "kpi"] = self.kpis_labels_mapping[ret.at[row, "kpi"]]
        ret["train"] = display_utils.fmt_apply(ret, "train", 0, "%")
        ret["backtest"] = display_utils.fmt_apply(ret, "backtest", 0, "%")
        ret.rename(columns=self.kpis_mapping, inplace=True)
        ret.set_index("kpi", inplace=True)
        return ret

    def display_calibration_vars(self, calibration_vars: pd.DataFrame):
        ret = calibration_vars.copy()
        calibration_criterion_list = []
        train_list = []
        backtest_list = []
        for row in range(ret.shape[0]):
            if ret.at[row, "calibration_criterion"] == "num_obs":
                train_list.append("{:,.0f}".format(ret.at[row, "train"]))
                backtest_list.append("{:,.0f}".format(ret.at[row, "backtest"]))
            else:
                train_list.append("{:,.2%}".format(ret.at[row, "train"]))
                backtest_list.append("{:,.2%}".format(ret.at[row, "backtest"]))
            calibration_criterion_list.append(
                self.calibration_labels_mapping[ret.at[row, "calibration_criterion"]])

        ret["calibration_criterion"] = calibration_criterion_list
        ret["train"] = train_list
        ret["train"].replace(["nan", "nan%"], "N/A", inplace=True)
        ret["backtest"] = backtest_list
        ret["backtest"].replace(["nan", "nan%"], "N/A", inplace=True)
        ret.rename(columns=self.calibration_mapping, inplace=True)
        ret.set_index("calibration_criterion", inplace=True)
        return ret

    def display_stability_vars(self, stability_vars: pd.DataFrame):
        ret = stability_vars[list(self.stability_mapping.keys())].drop_duplicates()
        ret.reset_index(drop=True, inplace=True)
        ret["psi_var"] = display_utils.fmt_apply(ret, "psi_var", 1, "%")
        ret.rename(columns=self.stability_mapping, inplace=True)
        return ret
    
    @staticmethod
    def display_ranking(ranking_df: pd.DataFrame):
        ranking_df = ranking_df.copy()
        lower_limit_list_main = \
            list(ranking_df.apply(lambda data_x: "{:,.2%}".format(data_x["lower_limit"]), axis=1))
        upper_limit_list_main = \
            list(ranking_df.apply(lambda data_x: "{:,.2%}".format(data_x["upper_limit"]), axis=1))
        bin_list = []
        target_rate_train_list = list(ranking_df["target_rate_train"])
        target_rate_backtest_list = list(ranking_df["target_rate_backtest"])
        n = len(lower_limit_list_main)
        for iteration in range(n):
            right_interval = ")"
            if iteration == n - 1:
                right_interval = "]"
            bin_list.append("[" + lower_limit_list_main[iteration] + ", " +
                            upper_limit_list_main[iteration] + right_interval)
        return bin_list, target_rate_train_list, target_rate_backtest_list

