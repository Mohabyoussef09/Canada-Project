"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import logging
import pandas as pd

from modellicity.extended_pandas.extended_properties.dataframe_properties import (
    DataFrameProperties,
)

logger = logging.getLogger(__name__)


class MultivariateUtils:
    """Multivariate analysis functions."""

    @staticmethod
    def decorrelate(
        dataframe: pd.DataFrame,
        kpi_map: pd.DataFrame,
        var_list: list,
        high_correlation_threshold: float,
    ) -> pd.DataFrame:
        """
        Exclude variables highly correlated with each other via a KPI.

        :param dataframe:
        :param kpi_map:
        :param var_list:
        :param high_correlation_threshold:
        :return:
        """
        remove_list = []
        props = DataFrameProperties()
        numeric_list_all = props.get_all_dataframe_numeric_object(dataframe)
        numeric_list = list(
            set(numeric_list_all)
            .intersection(set(var_list))
            .intersection(set(kpi_map["variable"]))
        )

        dataframe_numeric = dataframe[numeric_list].copy()

        dataframe_corr = dataframe_numeric.copy()
        dataframe_corr = dataframe_corr.corr()
        dataframe_corr = dataframe_corr.reset_index()
        dataframe_corr = dataframe_corr.rename(columns={"index": "Var1"})

        var1_list = []
        var2_list = []
        corr_list = []

        i = 0
        for var1 in dataframe_corr["Var1"]:
            j = 1
            for var2 in dataframe_corr["Var1"]:
                var1_list.append(var1)
                var2_list.append(var2)
                corr_list.append(dataframe_corr.iloc[i, j])
                j += 1
            i += 1

        dataframe_corr_long = pd.DataFrame()
        dataframe_corr_long["Var1"] = var1_list
        dataframe_corr_long["Var2"] = var2_list
        dataframe_corr_long["Corr"] = corr_list

        dataframe_corr_kpi = pd.merge(
            dataframe_corr_long,
            kpi_map,
            how="left",
            left_on="Var1",
            right_on="variable",
        )
        dataframe_corr_kpi.rename(columns={"kpi": "kpi1"}, inplace=True)
        # Remove "variable" as it is redundant to "Var1"
        dataframe_corr_kpi = dataframe_corr_kpi.drop(columns=["variable"])

        dataframe_corr_kpi = pd.merge(
            dataframe_corr_kpi, kpi_map, how="left", left_on="Var2", right_on="variable"
        )
        dataframe_corr_kpi = dataframe_corr_kpi.rename(columns={"kpi": "kpi2"})
        # Remove "variable" as it is redundant to "Var2"
        dataframe_corr_kpi = dataframe_corr_kpi.drop(columns=["variable"])

        # Sort by "kpi1" and "kpi2" highest to lowest as the loop below will
        # go from highest to lowest kpi value and decorrelate by removing every
        # var2 with its corresponding kpi2 less than var1's kpi1
        dataframe_corr_kpi = dataframe_corr_kpi.sort_values(
            by=["kpi1", "kpi2"], ascending=[False, False]
        )
        dataframe_corr_kpi = dataframe_corr_kpi.reset_index(drop=True)

        var1_loc = 0
        var2_loc = 1
        corr_loc = 2
        kpi1_loc = 3
        kpi2_loc = 4

        for i in range(len(dataframe_corr_kpi["Var1"])):
            var1 = dataframe_corr_kpi.iloc[i, var1_loc]
            var2 = dataframe_corr_kpi.iloc[i, var2_loc]
            corr = dataframe_corr_kpi.iloc[i, corr_loc]
            kpi1 = dataframe_corr_kpi.iloc[i, kpi1_loc]
            kpi2 = dataframe_corr_kpi.iloc[i, kpi2_loc]
            if var1 != var2 and abs(corr) > high_correlation_threshold and kpi1 > kpi2:
                remove_list.append(var2)

        remove_list = list(set(remove_list))
        dataframe_decorr = dataframe.copy()
        dataframe_decorr.drop(columns=remove_list, inplace=True)
        return dataframe_decorr
