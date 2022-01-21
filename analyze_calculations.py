from analyze_settings import IncludeExclude
from dataset import Dataset
from modellicity.src.modellicity.stats.weight_of_evidence import WeightOfEvidence
from artifacts import Artifacts
from settings import Settings

import numpy as np
import pandas as pd


def calculate_distribution_variables(data_df: pd.DataFrame) -> pd.DataFrame:
    """Generate dataframe to be used to output distribution analysis results."""
    ret = data_df.copy()

    percentile_values = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    percentile_keys = ["minimum", "perc_1", "perc_5", "perc_10", "perc_25", "perc_50",
                       "perc_75", "perc_90", "perc_95", "perc_99", "maximum"]

    ret = ret.quantile(percentile_values)
    ret.reset_index(inplace=True)
    ret["index"] = percentile_keys

    ret.rename(columns={"index": "variable"}, inplace=True)
    ret.set_index("variable", inplace=True)
    ret = ret.T

    ret.rename_axis(None, axis="columns", inplace=True)
    ret.reset_index(inplace=True)
    ret.rename(columns={"index": "variable"}, inplace=True)

    # Add "include_exclude" and "manually_set" and put them right after "variable"
    col_list = ["variable", "include_exclude_panel", "include_exclude", "manually_set"]
    col_list += [x for x in list(ret.columns) if x not in col_list]
    ret["include_exclude_panel"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["include_exclude"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["manually_set"] = [False] * ret.shape[0]
    ret = ret[col_list]

    return ret


def calculate_volatility_variables(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate dataframe to be used to output volatility analysis results.
    """
    df_ret = data_df.copy()

    # Std and mean
    df_std = pd.DataFrame(df_ret.std())
    df_std.rename(columns={0: "std"}, inplace=True)

    df_mean = pd.DataFrame(df_ret.mean())
    df_mean.rename(columns={0: "average"}, inplace=True)

    ret = pd.concat([df_std, df_mean], axis=1)
    ret.reset_index(inplace=True)
    ret.rename(columns={"index": "variable"}, inplace=True)

    # % most common value
    most_common_value_perc_list: list[float] = []
    n_obs = df_ret.shape[0]
    for var in df_ret.columns:
        most_common_value_perc_list.append(df_ret[var].value_counts().iloc[0] / n_obs)
    ret["perc_most_common_value"] = most_common_value_perc_list

    ret.sort_values(by=["perc_most_common_value", "variable"],
                    ascending=[True, True],
                    inplace=True)
    ret.reset_index(drop=True, inplace=True)

    # Add "include_exclude" and "manually_set" and put them right after "variable"
    col_list = ["variable", "include_exclude_panel", "include_exclude", "manually_set"]
    col_list += [x for x in list(ret.columns) if x not in col_list]
    ret["include_exclude_panel"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["include_exclude"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["manually_set"] = [False] * ret.shape[0]
    ret = ret[col_list]

    return ret


def calculate_correlation_variables(input_df: pd.DataFrame, target_label: str = None,
                                    bin_vars_dict: dict[str, float] = None) -> pd.DataFrame:
    """
    Calculate information value (IV) using Modellicity's inbuilt library. First, a variable is
    checked if its IV has already been calculated during the "Bin variables" stage; if so, the
    value is used; if not, the Modellicity library is used with default settings. Correlation
    is the sample Pearson correlation. For values that are one-valued, the correlation values
    with other values are set to zero.
    """
    df = input_df.copy()
    if target_label is not None:
        ret = df.drop(columns=target_label)
    else:
        ret = df.copy()

    col_list = list(ret.columns)
    df_iv = pd.DataFrame()
    if target_label is not None:
        # Extract the variables whose IVs have been calculated before and their corresponding IV
        bin_already_list: list[str] = []
        iv_already_list: list[float] = []
        if bin_vars_dict is not None:
            bin_already_list = [x for x in col_list if x in list(bin_vars_dict.keys())]
            iv_already_list = [bin_vars_dict[x] for x in bin_already_list]

        # Extract the variables whose IVs have not been calculated before and calculate their IV
        bin_first_time_list = [var for var in col_list if var not in bin_already_list]
        iv_first_time_list = []
        if len(bin_first_time_list) > 0:
            woe = WeightOfEvidence()
            df_iv_first_time = \
                woe.calculate_woe_iv_from_df(df=df,
                                             var_list=bin_first_time_list,
                                             target_label=target_label)
            df_iv_first_time = df_iv_first_time[["variable", "iv"]].drop_duplicates("variable")

            iv_first_time_list = df_iv_first_time["iv"].tolist()

        df_iv["variable"] = bin_already_list + df_iv_first_time["variable"].tolist()
        df_iv["iv"] = iv_already_list + df_iv_first_time["iv"].tolist()
    else:
        df_iv["variable"] = col_list
        df_iv["iv"] = [np.nan] * len(col_list)

    df_iv.sort_values(by=["iv", "variable"], ascending=[False, True], inplace=True)
    df_iv.reset_index(drop=True, inplace=True)

    # Calculate correlations and put in same order as IV
    df_corr_matrix = df[col_list].corr()

    # Address the case where there are one-valued variables and Pandas sets them to NULL
    df_corr_matrix.fillna(0, inplace=True)
    for i in range(df_corr_matrix.shape[0]):
        df_corr_matrix.iloc[i, i] = 1.0

    df_corr_matrix.reset_index(inplace=True)
    df_corr_matrix.rename(columns={"index": "variable"}, inplace=True)
    df_corr_var_list = ["variable"]
    df_corr_var_list.extend(list(df_iv["variable"]))
    df_corr_matrix = df_corr_matrix[df_corr_var_list]

    ret = pd.merge(df_iv, df_corr_matrix, how="left", on="variable")
    # Set upper diagonal values to NULL
    for i in range(ret.shape[0]):
        for j in range(i + 3, ret.shape[1]):
            ret.iloc[i, j] = np.nan

    # Add "include_exclude" and "manually_set" and put them right after "variable"
    col_list = ["variable", "include_exclude_panel", "include_exclude", "manually_set"]
    col_list += [x for x in list(ret.columns) if x not in col_list]
    ret["include_exclude_panel"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["include_exclude"] = [IncludeExclude.INCLUDE] * ret.shape[0]
    ret["manually_set"] = [False] * ret.shape[0]
    ret = ret[col_list]

    return ret


class AnalyzeCalculate:
    def __init__(self, dataset: Dataset, settings: Settings, artifacts: Artifacts):
        self.ds = dataset

        self.analyze_settings = settings.analyze_settings
        self.binning_artifacts = artifacts.binning_artifacts

        self.bin_vars_dict = self.binning_artifacts.binned_vars_dict
        self.binned_list: list[str] = []

        if self.binning_artifacts.bin_vars_df is not None:
            self.binned_list = list(self.binning_artifacts.bin_vars_df["variable"].unique())

        self.distribution_vars = calculate_distribution_variables(
            self.ds.data[self.ds.model_qualified_var_labels])
        self.volatility_vars = calculate_volatility_variables(
            self.ds.data[self.ds.model_qualified_var_labels])

        if self.ds.target_label is None:
            self.correlation_vars = calculate_correlation_variables(
                self.ds.data[self.ds.model_qualified_var_labels],
                target_label= None,
                bin_vars_dict=self.bin_vars_dict)
        else:
            self.correlation_vars = calculate_correlation_variables(
                self.ds.data[self.ds.model_qualified_var_labels + [self.ds.target_label]],
                target_label=self.ds.target_label,
                bin_vars_dict=self.bin_vars_dict)

        self.disabled_var_labels: list[str] = []
        if self.analyze_settings.include.binned_vars_only:
            self.disabled_var_labels.extend([x for x in self.ds.model_qualified_var_labels
                                             if x not in self.binned_list])

        self.volatility_flagged_vars = self.get_volatility_flagged_vars()
        self.correlation_flagged_vars = self.get_correlation_flagged_vars()
        # All calculations up to this point are panel based, so we make a copy
        self.disabled_var_labels_panel = self.disabled_var_labels.copy()

        self.add_settings_results()

    def get_volatility_flagged_vars(self) -> list[tuple[int, int]]:
        """
        Generate list of variable names and corresponding index coordinates that will be used to be
        flagged in red areas of volatility analysis table.
        """
        flagged_indices: list[tuple[int, int]] = []
        # Column coordinate.
        j = 6
        for i in range(self.volatility_vars.shape[0]):
            if self.volatility_vars.iloc[i, j] >= \
                    self.analyze_settings.include.perc_most_common_lt / 100.0:
                flagged_indices.append((i, j))
                if self.volatility_vars["variable"].iloc[i] not in self.disabled_var_labels:
                    self.disabled_var_labels.append(self.volatility_vars["variable"].iloc[i])

        return flagged_indices

    def get_correlation_flagged_vars(self) -> list[tuple[int, int]]:
        """
        Generate list of variable names and corresponding index coordinates that will be used to
        be flagged in red areas of correlation analysis table.
        """
        flagged_indices: list[tuple[int, int]] = []
        for i in range(self.correlation_vars.shape[0]):
            if self.correlation_vars.iloc[i, 4] < \
                    self.analyze_settings.include.information_value_gt / 100.0:
                flagged_indices.append((i, 4))
                # Append to self.disabled_var_labels
                if self.correlation_vars["variable"].iloc[i] not in self.disabled_var_labels:
                    self.disabled_var_labels.append(self.correlation_vars["variable"].iloc[i])

            for j in range(5, min(i + 5, self.correlation_vars.shape[1])):
                if abs(self.correlation_vars.iloc[i, j]) >= \
                        self.analyze_settings.include.abs_correlation_lt / 100.0:
                    flagged_indices.append((i, j))
                    # Append to self.disabled_var_labels
                    if self.correlation_vars["variable"].iloc[i] not in self.disabled_var_labels:
                        self.disabled_var_labels.append(self.correlation_vars["variable"].iloc[i])
        return flagged_indices

    def add_settings_results(self):
        self.distribution_vars["include_exclude_panel"] = \
            np.select([self.distribution_vars["variable"].isin(self.disabled_var_labels_panel),
                       ~(self.distribution_vars["variable"].isin(self.disabled_var_labels_panel))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])
        self.distribution_vars["include_exclude"] = \
            np.select([self.distribution_vars["variable"].isin(self.disabled_var_labels),
                       ~(self.distribution_vars["variable"].isin(self.disabled_var_labels))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])

        self.volatility_vars["include_exclude_panel"] = \
            np.select([self.volatility_vars["variable"].isin(self.disabled_var_labels_panel),
                       ~(self.volatility_vars["variable"].isin(self.disabled_var_labels_panel))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])
        self.volatility_vars["include_exclude"] = \
            np.select([self.volatility_vars["variable"].isin(self.disabled_var_labels),
                       ~(self.volatility_vars["variable"].isin(self.disabled_var_labels))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])

        self.correlation_vars["include_exclude_panel"] = \
            np.select([self.correlation_vars["variable"].isin(self.disabled_var_labels_panel),
                       ~(self.correlation_vars["variable"].isin(self.disabled_var_labels_panel))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])
        self.correlation_vars["include_exclude"] = \
            np.select([self.correlation_vars["variable"].isin(self.disabled_var_labels),
                       ~(self.correlation_vars["variable"].isin(self.disabled_var_labels))],
                      [IncludeExclude.EXCLUDE, IncludeExclude.INCLUDE])

    def update_tables_from_settings(self):
        self.disabled_var_labels: list[str] = []
        if self.analyze_settings.include.binned_vars_only:
            self.disabled_var_labels.extend([x for x in self.ds.model_qualified_var_labels
                                             if x not in self.binned_list])
        self.distribution_vars["manually_set"] = [False] * self.distribution_vars.shape[0]
        self.volatility_vars["manually_set"] = [False] * self.volatility_vars.shape[0]
        self.correlation_vars["manually_set"] = [False] * self.correlation_vars.shape[0]
        self.volatility_flagged_vars = self.get_volatility_flagged_vars()
        self.correlation_flagged_vars = self.get_correlation_flagged_vars()
        # All calculations up to this point are panel based, so we make a copy
        self.disabled_var_labels_panel = self.disabled_var_labels.copy()
        self.add_settings_results()
