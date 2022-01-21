import pandas as pd
import display_utils


class AnalyzeDisplay:
    def __init__(self):
        self.distribution_display_column_list = [
            "variable",
            "include_exclude",
            "manually_set",
            "minimum",
            "perc_1",
            "perc_5",
            "perc_10",
            "perc_25",
            "perc_50",
            "perc_75",
            "perc_90",
            "perc_95",
            "perc_99",
            "maximum",
        ]
        self.distribution_label_mapping = {
            "variable": "Variable",
            "include_exclude": "Include/Exclude",
            "manually_set": "Manual/Panel",
            "minimum": "Minimum",
            "perc_1": "1 percentile",
            "perc_5": "5 percentile",
            "perc_10": "10 percentile",
            "perc_25": "25 percentile",
            "perc_50": "50 percentile",
            "perc_75": "75 percentile",
            "perc_90": "90 percentile",
            "perc_95": "95 percentile",
            "perc_99": "99 percentile",
            "maximum": "Maximum",
        }

        self.volatility_display_column_list = [
            "variable",
            "include_exclude",
            "manually_set",
            "std",
            "average",
            "perc_most_common_value",
        ]

        self.volatility_label_mapping = {
            "variable": "Variable",
            "include_exclude": "Include/Exclude",
            "manually_set": "Manual/Panel",
            "std": "Std",
            "average": "Average",
            "perc_most_common_value": "% most common value",
        }

        self.correlation_display_exclude_column_list = [
            "include_exclude_panel"
        ]

        self.correlation_label_mapping = {
            "variable": "Variable",
            "include_exclude": "Include/Exclude",
            "manually_set": "Manual/Panel",
            "iv": "IV",
        }

    def display_distribution_variables(self, distribution_vars: pd.DataFrame) -> pd.DataFrame:
        ret = distribution_vars[self.distribution_display_column_list].copy()

        ignore_cols = {"variable", "include_exclude", "manually_set"}
        dist_cols = list(set(list(ret.columns)) - ignore_cols)

        for _, col in enumerate(dist_cols):
            ret[col] = display_utils.fmt_apply(ret, col, 2, "f")

        ret["include_exclude"] = [str(x) for x in list(ret["include_exclude"])]
        ret["manually_set"] = ["Manual" if x else "Panel" for x in list(ret["manually_set"])]

        ret.rename(columns=self.distribution_label_mapping, inplace=True)
        return ret

    def display_volatility_variables(self, volatility_vars: pd.DataFrame) -> pd.DataFrame:
        ret = volatility_vars[self.volatility_display_column_list].copy()

        ret["std"] = display_utils.fmt_apply(ret, "std", 2, "f")
        ret["average"] = display_utils.fmt_apply(ret, "average", 2, "f")
        ret["perc_most_common_value"] = display_utils.fmt_apply(ret, "perc_most_common_value", 0, "%")

        ret["include_exclude"] = [str(x) for x in list(ret["include_exclude"])]
        ret["manually_set"] = ["Manual" if x else "Panel" for x in list(ret["manually_set"])]

        ret.rename(columns=self.volatility_label_mapping, inplace=True)
        return ret

    def display_correlation_variables(self, correlation_vars: pd.DataFrame) -> pd.DataFrame:
        ret = correlation_vars.drop(columns=self.correlation_display_exclude_column_list).copy()

        ignore_cols = {"variable", "include_exclude", "manually_set"}
        corr_cols = list(set(list(ret.columns)) - ignore_cols)

        for _, col in enumerate(corr_cols):
            ret[col] = display_utils.fmt_apply(ret, col, 1, "%")
            ret[col] = ret[col].replace("nan%", "")

        ret["include_exclude"] = [str(x) for x in list(ret["include_exclude"])]
        ret["manually_set"] = ["Manual" if x else "Panel" for x in list(ret["manually_set"])]

        ret.rename(columns=self.correlation_label_mapping, inplace=True)
        return ret

    @staticmethod
    def get_volatility_flagged_display(flagged_vars) -> list[tuple[int, int]]:
        flagged_indices: list[tuple[int, int]] = []
        for i in range(len(flagged_vars)):
            flagged_indices.append((flagged_vars[i][0], flagged_vars[i][1] - 1))
        return flagged_indices

    @staticmethod
    def get_correlation_flagged_display(flagged_vars) -> list[tuple[int, int]]:
        flagged_indices: list[tuple[int, int]] = []
        for i in range(len(flagged_vars)):
            flagged_indices.append((flagged_vars[i][0], flagged_vars[i][1] - 1))
        return flagged_indices