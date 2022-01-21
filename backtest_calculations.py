from artifacts import Artifacts
from dataset import Dataset
from scipy import stats
from settings import Settings
from sklearn.metrics import roc_auc_score
from typing import Any
import numpy as np
import pandas as pd


class BacktestCalculate:
    def __init__(self, dataset: Dataset, settings: Settings, artifacts: Artifacts):
        self.ds = dataset
        self.settings = settings  # ToDO: Consider removing if not to be used even after scaling
        self.artifacts = artifacts

    @property
    def var_list(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.artifacts.train_artifacts.model.variable_list

    @property
    def target_label(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.ds.target_label

    @property
    def probability_label(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.ds.probability_label

    @property
    def data_train(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.artifacts.train_artifacts.model.data_x[self.var_list]

    @property
    def target_train(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.artifacts.train_artifacts.model.target_y

    @property
    def probability_train(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.artifacts.train_artifacts.model.prob_y

    @property
    def data_backtest(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            not_found_list = [x for x in self.var_list if x not in self.ds.data.columns]
            if len(not_found_list) > 0:
                return None
            return self.ds.data[self.var_list]

    @property
    def target_backtest(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.ds.data[self.target_label]

    @property
    def probability_backtest(self):
        if self.artifacts.train_artifacts.all_artifacts_available:
            return self.ds.data[self.probability_label]

    def calculate_kpis_vars(self):
        if not self.artifacts.train_artifacts.all_artifacts_available:
            return None
        tmp_train = self.artifacts.train_artifacts.kpis_df
        auroc_train = tmp_train[tmp_train["kpi"] == "auroc"]["value"].iloc[0]
        ks_train = tmp_train[tmp_train["kpi"] == "ks"]["value"].iloc[0]

        tmp_backtest = pd.DataFrame()
        tmp_backtest["target"], tmp_backtest["prob"] = \
            self.target_backtest, self.probability_backtest
        tmp_backtest_event = tmp_backtest[tmp_backtest["target"] == 1]
        tmp_backtest_non_event = tmp_backtest[tmp_backtest["target"] == 0]
        auroc_backtest = roc_auc_score(self.target_backtest, self.probability_backtest)
        ks_backtest = stats.ks_2samp(tmp_backtest_event["prob"], tmp_backtest_non_event["prob"])[0]

        return pd.DataFrame({"kpi": ["auroc", "ks"],
                             "train": [auroc_train, ks_train],
                             "backtest": [auroc_backtest, ks_backtest]})

    @staticmethod
    def calculate_target_rate(target_y: pd.Series):
        return target_y.mean()

    @staticmethod
    def calculate_average_probability(prob_y: pd.Series):
        return prob_y.mean()

    def calculate_calibration_vars(self):
        if not self.artifacts.train_artifacts.all_artifacts_available:
            return None
        tmp_train = self.artifacts.train_artifacts.calibration_df.copy()
        num_obs_train = self.artifacts.train_artifacts.model.data_x.shape[0]
        num_obs_backtest = self.ds.data.shape[0]
        target_rate_train = \
            tmp_train[tmp_train["calibration_criterion"] == "target_rate"]["value"].iloc[0]
        average_probability_train = \
            tmp_train[tmp_train["calibration_criterion"] == "average_probability"]["value"].iloc[0]
        difference_train = \
            tmp_train[tmp_train["calibration_criterion"] == "difference"]["value"].iloc[0]
        percentage_change_train = \
            tmp_train[tmp_train["calibration_criterion"] == "percentage_change"]["value"].iloc[0]

        target_rate_backtest = self.calculate_target_rate(self.target_backtest)
        average_probability_backtest = self.calculate_average_probability(self.probability_backtest)
        difference_backtest = average_probability_backtest - target_rate_backtest
        percentage_change_backtest = difference_backtest / target_rate_backtest

        return pd.DataFrame({
            "calibration_criterion": [
                "num_obs",
                "target_rate", "average_probability", "difference",
                "percentage_change"],
            "train": [num_obs_train,
                      target_rate_train,
                      average_probability_train,
                      difference_train,
                      percentage_change_train],
            "backtest": [num_obs_backtest,
                         target_rate_backtest,
                         average_probability_backtest,
                         difference_backtest,
                         percentage_change_backtest]}
        )

    @staticmethod
    def get_psi_var(main_input: pd.Series, benchmark_input: pd.Series,
                    percentiles_input: int = 10, var_label: str = None) -> \
            pd.DataFrame:
        """Objective: Get PSI information for a variable."""
        n = percentiles_input
        percentile_values = [i / n * 100 for i in range(n + 1)]
        percentile_list = list(pd.Series(np.percentile(main_input, percentile_values)).unique())
        if len(percentile_list) <= 2:
            percentile_list.append(max(percentile_list)+1e-5)
        n = len(percentile_list) - 1
        lower_limit_list = percentile_list[0:n]
        upper_limit_list = percentile_list[1:n + 1]

        num_obs_x = len(main_input)
        num_obs_y = len(benchmark_input)
        bin_num_list = []
        bin_list = []
        concentration_x_list = []
        concentration_y_list = []
        psi_bin_list = []
        psi_var = 0
        df_temp = pd.DataFrame({"x": main_input, "y": benchmark_input})
        for i in range(n):
            bin_num_list.append(i+1)
            df_temp_x_var = df_temp[df_temp["x"] >= lower_limit_list[i]]
            df_temp_y_var = df_temp[df_temp["y"] >= lower_limit_list[i]]
            if i == n - 1:
                bin_list.append(
                    "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + "]")
                df_temp_x_var = df_temp_x_var[df_temp_x_var["x"] <= upper_limit_list[i]]
                df_temp_y_var = df_temp_y_var[df_temp_y_var["y"] <= upper_limit_list[i]]
            else:
                bin_list.append(
                    "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + ")")
                df_temp_x_var = df_temp_x_var[df_temp_x_var["x"] < upper_limit_list[i]]
                df_temp_y_var = df_temp_y_var[df_temp_y_var["y"] < upper_limit_list[i]]

            concentration_x_list.append(df_temp_x_var.shape[0] / num_obs_x)
            concentration_y_list.append(df_temp_y_var.shape[0] / num_obs_y)

            if concentration_y_list[i] == 0:
                psi_bin_list.append(0)
            else:
                psi_bin_list.append((concentration_x_list[i] - concentration_y_list[i]) *
                                    np.log(concentration_x_list[i] / concentration_y_list[i]))
            psi_var += psi_bin_list[i]

        psi_var_df = pd.DataFrame()
        if var_label is not None:
            psi_var_df["variable"] = [var_label] * n

        psi_var_df["bin_num"] = bin_num_list
        psi_var_df["lower_limit"] = lower_limit_list
        psi_var_df["upper_limit"] = upper_limit_list
        psi_var_df["bin"] = bin_list
        psi_var_df["concentration_x"] = concentration_x_list
        psi_var_df["concentration_y"] = concentration_y_list
        psi_var_df["psi_bin"] = psi_bin_list
        psi_var_df["psi_var"] = [psi_var] * n

        return psi_var_df

    def calculate_stability_vars(self):
        if not self.artifacts.train_artifacts.all_artifacts_available:
            return None

        psi_prob_df = self.get_psi_var(self.probability_train,
                                       self.probability_backtest,
                                       var_label=self.probability_label)
        psi_vars_df = pd.DataFrame()
        for var in self.var_list:
            psi_var_df = self.get_psi_var(self.data_train[var], self.data_backtest[var], var_label=var)
            psi_vars_df = pd.concat([psi_vars_df, psi_var_df], ignore_index=True)
        psi_vars_df.sort_values(by=["psi_var", "variable", "bin_num"],
                                ascending=[False, True, True], inplace=True)
        psi_vars_df.reset_index(drop=True, inplace=True)
        return pd.concat([psi_prob_df, psi_vars_df], ignore_index=True)

    def calculate_ranking_vars(self):
        y_train_input, y_backtest_input = self.target_train, self.target_backtest
        y_train_prob_input, y_backtest_prob_input = self.probability_train, self.probability_backtest
        percentiles_input = 5

        df_temp_train = pd.DataFrame({"y": y_train_input, "y_prob": y_train_prob_input})
        df_temp_backtest = pd.DataFrame({"y": y_backtest_input, "y_prob": y_backtest_prob_input})
        rank_output_df: dict[str, Any] = {
            "bin_num": [],
            "lower_limit": [],
            "upper_limit": [],
            "bin": [],
            "num_obs_train": [],
            "num_events_train": [],
            "target_rate_train": [],
            "num_obs_backtest": [],
            "num_events_backtest": [],
            "target_rate_backtest": []
            }

        n = percentiles_input
        percentile_values = [i / n * 100 for i in range(n + 1)]
        percentile_list = list(
            pd.Series(np.percentile(y_train_prob_input, percentile_values)).unique())
        percentile_list[0] = 0  # Replace the minimum value with 0
        n = len(percentile_list) - 1
        percentile_list[n] = 1  # Replace the maximum value with 1
        lower_limit_list = percentile_list[0:n]
        upper_limit_list = percentile_list[1:n + 1]
        rank_output_df["bin_num"] = list(range(1, n + 1))
        rank_output_df["lower_limit"] = lower_limit_list
        rank_output_df["upper_limit"] = upper_limit_list
        for i in range(n):
            df_temp_train_var = df_temp_train[df_temp_train["y_prob"] >= lower_limit_list[i]]
            df_temp_backtest_var = df_temp_backtest[
                df_temp_backtest["y_prob"] >= lower_limit_list[i]]
            if i == n - 1:
                bin_value = "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + "]"
                df_temp_train_var = \
                    df_temp_train_var[df_temp_train_var["y_prob"] <= upper_limit_list[i]]
                df_temp_backtest_var = \
                    df_temp_backtest_var[df_temp_backtest_var["y_prob"] <= upper_limit_list[i]]
            else:
                bin_value = "[" + str(lower_limit_list[i]) + ", " + str(upper_limit_list[i]) + ")"
                df_temp_train_var = df_temp_train_var[df_temp_train_var["y_prob"] < upper_limit_list[i]]
                df_temp_backtest_var = df_temp_backtest_var[df_temp_backtest_var["y_prob"] < upper_limit_list[i]]

            num_obs_train, num_obs_backtest = df_temp_train_var.shape[0], df_temp_backtest_var.shape[0]
            num_events_train, num_events_backtest = df_temp_train_var["y"].sum(), df_temp_backtest_var["y"].sum()

            target_rate_train_value, target_rate_backtest_value = 0, 0
            if num_obs_train != 0:
                target_rate_train_value = num_events_train / num_obs_train
            if num_obs_backtest != 0:
                target_rate_backtest_value = num_events_backtest / num_obs_backtest

            rank_output_df["bin"].append(bin_value)
            rank_output_df["num_obs_train"].append(num_obs_train)
            rank_output_df["num_events_train"].append(num_events_train)
            rank_output_df["target_rate_train"].append(target_rate_train_value)
            rank_output_df["num_obs_backtest"].append(num_obs_backtest)
            rank_output_df["num_events_backtest"].append(num_events_backtest)
            rank_output_df["target_rate_backtest"].append(target_rate_backtest_value)

        return pd.DataFrame().from_dict(rank_output_df)

    def calculate_distribution_vars(self):
        train_input = pd.DataFrame()
        train_input[self.probability_label] = self.probability_train
        train_input = pd.concat([train_input, self.data_train], axis=1)

        backtest_input = pd.DataFrame()
        backtest_input[self.probability_label] = self.probability_backtest
        backtest_input = pd.concat([backtest_input, self.data_backtest], axis=1)

        dis_data_df = pd.DataFrame()
        dis_data_df["variable"] = train_input.columns
        dis_data_df["minimum_train"] = list(train_input.min())
        dis_data_df["minimum_backtest"] = list(backtest_input.min())
        dis_data_df["percentile_5_train"] = list(train_input.quantile(0.05))
        dis_data_df["percentile_5_backtest"] = list(backtest_input.quantile(0.05))
        dis_data_df["percentile_50_train"] = list(train_input.quantile(0.50))
        dis_data_df["percentile_50_backtest"] = list(backtest_input.quantile(0.50))
        dis_data_df["average_train"] = list(train_input.mean())
        dis_data_df["average_backtest"] = list(backtest_input.mean())
        dis_data_df["percentile_95_train"] = list(train_input.quantile(0.95))
        dis_data_df["percentile_95_backtest"] = list(backtest_input.quantile(0.95))
        dis_data_df["maximum_train"] = list(train_input.max())
        dis_data_df["maximum_backtest"] = list(backtest_input.max())
        return dis_data_df
