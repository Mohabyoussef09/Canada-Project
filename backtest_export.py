import pandas as pd


class BacktestExport:
    def __init__(self):
        self.kpis_mapping = {
            "kpi": "KPI",
            "train": "Training",
            "backtest": "Backtesting",
        }
        self.kpis_labels_mapping = {
            "auroc": "AUROC",
            "ks": "KS",
        }

        self.calibration_mapping = {
            "calibration_criterion": "Calibration criterion",
            "train": "Training",
            "backtest": "Backtesting",
        }
        self.calibration_labels_mapping = {
            "num_obs": "# observations",
            "calibration_setting": "Calibration setting",
            "target_rate": "Target rate",
            "average_probability": "Average probability",
            "difference": "Difference",
            "percentage_change": "Percentage change",
        }

        self.stability_mapping = {
            "variable": "Variable",
            "bin_num": "Bin #",
            "lower_limit": "Lower limit",
            "upper_limit": "Upper limit",
            "bin": "Bin",
            "concentration_x": "Concentration % (training)",
            "concentration_y": "Concentration % (backtesting)",
            "psi_bin": "PSI (bin level)",
            "psi_var": "PSI (variable level)",
        }

        self.ranking_mapping = {
            "bin_num": "Bin #",
            "lower_limit": "Lower limit",
            "upper_limit": "Upper limit",
            "bin": "Bin",
            "num_obs_train": "# obs (training)",
            "num_events_train": "# events (training)",
            "target_rate_train": "Target rate (training)",
            "num_obs_backtest": "# obs (backtesting)",
            "num_events_backtest": "# events (backtesting)",
            "target_rate_backtest": "Target rate (backtesting)",
        }

        self.distribution_mapping = {
            "variable": "Variable",
            "minimum_train": "Minimum (training)",
            "minimum_backtest": "Minimum (backtesting)",
            "percentile_5_train": "5 percentile (training)",
            "percentile_5_backtest": "5 percentile (backtesting)",
            "percentile_50_train": "50 percentile (training)",
            "percentile_50_backtest": "50 percentile (backtesting)",
            "average_train": "Average (training)",
            "average_backtest": "Average (backtesting)",
            "percentile_95_train": "95 percentile (training)",
            "percentile_95_backtest": "95 percentile (backtesting)",
            "maximum_train": "Maximum (training)",
            "maximum_backtest": "Maximum (backtesting)",
        }

    def export_kpis_vars(self, kpis_vars: pd.DataFrame):
        ret = kpis_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "kpi"] = self.kpis_labels_mapping[ret.at[row, "kpi"]]
        ret.rename(columns=self.kpis_mapping, inplace=True)
        return ret

    def export_calibration_vars(self, calibration_vars: pd.DataFrame):
        ret = calibration_vars.copy()
        for row in range(ret.shape[0]):
            ret.at[row, "calibration_criterion"] = \
                self.calibration_labels_mapping[ret.at[row, "calibration_criterion"]]
        ret.rename(columns=self.calibration_mapping, inplace=True)
        return ret

    def export_stability_vars(self, stability_vars: pd.DataFrame):
        ret = stability_vars.copy()
        ret.rename(columns=self.stability_mapping, inplace=True)
        return ret

    def export_ranking_vars(self, ranking_vars: pd.DataFrame):
        ret = ranking_vars.copy()
        ret.rename(columns=self.ranking_mapping, inplace=True)
        return ret

    def export_distribution_vars(self, distribution_vars: pd.DataFrame):
        ret = distribution_vars.copy()
        ret.rename(columns=self.distribution_mapping, inplace=True)
        return ret

    def export_all(self,
                   kpis_vars: pd.DataFrame,
                   calibration_vars: pd.DataFrame,
                   stability_vars: pd.DataFrame,
                   ranking_vars: pd.DataFrame,
                   distribution_vars: pd.DataFrame):
        df_filler = pd.DataFrame()
        df_filler[""] = [""] * 1
        df1 = self.export_kpis_vars(kpis_vars)
        df2 = self.export_calibration_vars(calibration_vars)
        df3 = self.export_stability_vars(stability_vars)
        df4 = self.export_ranking_vars(ranking_vars)
        df5 = self.export_distribution_vars(distribution_vars)
        ret = pd.concat(
            [df1, df_filler,
             df2, df_filler,
             df3, df_filler,
             df4, df_filler,
             df5],
            axis=1,
        )
        return ret
