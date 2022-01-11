from binning_settings import BinAsMode, BinningNumericMode, BinningSettings
from modellicity.src.modellicity.stats.weight_of_evidence import WeightOfEvidence
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
import df_utils


def bin_as_categorical(settings: BinningSettings, perc_missing: float, num_unique_values: int):
    if perc_missing >= settings.exclude.perc_missing_vals_gt / 100.0:
        return BinAsMode.NONE

    if num_unique_values < settings.exclude.num_unique_vals_lt:
        return BinAsMode.NONE

    if settings.categorical.bin_all_checkbox:
        return BinAsMode.CATEGORICAL
    return BinAsMode.NONE


def bin_as_datetime(settings: BinningSettings, perc_missing: float, num_unique_values: int):
    if perc_missing >= settings.exclude.perc_missing_vals_gt / 100.0:
        return BinAsMode.NONE

    if num_unique_values < settings.exclude.num_unique_vals_lt:
        return BinAsMode.NONE

    if settings.exclude.datetime_vars:
        return BinAsMode.NONE
    return BinAsMode.CATEGORICAL


def bin_as_numeric(settings: BinningSettings, perc_missing: float, num_unique_values: int):
    if perc_missing >= settings.exclude.perc_missing_vals_gt / 100.0:
        return BinAsMode.NONE

    if num_unique_values < settings.exclude.num_unique_vals_lt:
        return BinAsMode.NONE

    if settings.numeric.bin_mode == BinningNumericMode.ALL or \
            (settings.numeric.bin_mode == BinningNumericMode.ONLY_WITH_MISSING and perc_missing > 0):
        if not settings.numeric.lower_bound_is_infinity and \
                not settings.numeric.upper_bound_is_infinity:
            return BinAsMode.NUM_MIN_MAX
        if settings.numeric.lower_bound_is_infinity and \
                not settings.numeric.upper_bound_is_infinity:
            return BinAsMode.NUM_MINUS_INF_MAX
        if not settings.numeric.lower_bound_is_infinity and \
                settings.numeric.upper_bound_is_infinity:
            return BinAsMode.NUM_MIN_PLUS_INF
        return BinAsMode.NUM_MINUS_INF_PLUS_INF
    return BinAsMode.NONE


def calculate_flagged_data_vars(df_input: pd.DataFrame) -> list[int]:
    return df_input[df_input["bin_as"] == BinAsMode.NONE].index.tolist()


def setup_binning_dataset(target_label_input: str, bin_var_list_input: list[str],
                          bin_as_list_input: list[BinAsMode], df_input: pd.DataFrame) -> pd.DataFrame:
    """Setup dataset to be fed into WOE function."""
    ret = df_input[bin_var_list_input].copy()
    if target_label_input:
        ret[target_label_input] = df_input[target_label_input].copy()

    for i, var in enumerate(bin_var_list_input):
        if bin_as_list_input[i] is BinAsMode.CATEGORICAL and not is_string_dtype(ret[var]):
            ret[var] = np.select([ret[var].isin([np.nan]), ~ret[var].isin([np.nan])], [np.nan, ret[var].astype(str)])
        elif bin_as_list_input[i] in [BinAsMode.NUM_MIN_MAX,
                                      BinAsMode.NUM_MINUS_INF_MAX,
                                      BinAsMode.NUM_MIN_PLUS_INF,
                                      BinAsMode.NUM_MINUS_INF_PLUS_INF] and not is_numeric_dtype(ret[var]):
            ret[var] = pd.to_numeric(ret[var])

    return ret


def setup_binning_lower_bound(var_series_input: pd.Series, bin_as_series_input: pd.Series) -> dict[str, bool]:
    """Determine lower bounds of variables."""
    ret = np.select([bin_as_series_input.isin([BinAsMode.NUM_MINUS_INF_MAX, BinAsMode.NUM_MINUS_INF_PLUS_INF]),
                     ~bin_as_series_input.isin([BinAsMode.NUM_MINUS_INF_MAX, BinAsMode.NUM_MINUS_INF_PLUS_INF])],
                    [False, True])
    return dict(zip(var_series_input, ret))


def setup_binning_upper_bound(var_series_input: pd.Series, bin_as_series_input: pd.Series) -> dict[str, bool]:
    """Determine upper bounds of variables."""
    ret = np.select([bin_as_series_input.isin([BinAsMode.NUM_MIN_PLUS_INF, BinAsMode.NUM_MINUS_INF_PLUS_INF]),
                     ~bin_as_series_input.isin([BinAsMode.NUM_MIN_PLUS_INF, BinAsMode.NUM_MINUS_INF_PLUS_INF])],
                    [False, True])
    return dict(zip(var_series_input, ret))


def apply_binning(df_input: pd.DataFrame, df_woe_map: pd.DataFrame) -> pd.DataFrame:
    """Apply binning on a dataframe as per a mapping scheme"""
    woe = WeightOfEvidence()
    bin_var_parent_list = list(df_woe_map["variable_parent"].unique())
    bin_var_child_list = list(df_woe_map["variable"].unique())
    ret = df_input.copy()
    for i, var_parent in enumerate(bin_var_parent_list):
        var_child = bin_var_child_list[i]
        df_woe_map_var = df_woe_map[df_woe_map["variable_parent"] == var_parent]
        ret[var_child] = woe.map_to_woe(variable=ret[var_parent], df_woe_map=df_woe_map_var)

    return ret


def calculate_data_variables(df_input: pd.DataFrame, settings: BinningSettings) -> pd.DataFrame:
    """Return data variables table before any user changes."""
    df_datetime = df_utils.get_all_datetime_format(df_input)
    df_missing = df_utils.get_percent_missing(df_input)
    df_types = df_utils.get_variable_types(df_input)
    df_num_unique_values = df_utils.get_num_unique_values(df_input)

    data_variables = {
        "variable": df_input.columns.tolist(),
        "type": [],
        "perc_missing": [],
        "num_unique_values": [],
        "bin_as": [],
        "bin_as_panel": [],
        "manually_set": [False] * df_input.shape[1],
    }
    for variable in data_variables["variable"]:
        if variable in df_datetime[df_datetime["is_datetime_format"] == True]["variable"].tolist():
            variable_type = "Datetime"
        else:
            variable_type = df_types.loc[variable, "type"]
        data_variables["type"].append(variable_type)

        perc_missing = df_missing.loc[variable, "perc_missing"]
        data_variables["perc_missing"].append(perc_missing)

        num_unique_values = df_num_unique_values.loc[variable, "num_unique_values"]
        data_variables["num_unique_values"].append(num_unique_values)

        if variable_type == "Datetime":
            data_variables["bin_as"].append(bin_as_datetime(settings, perc_missing,
                                                            num_unique_values))
        elif variable_type == "Numeric":
            data_variables["bin_as"].append(bin_as_numeric(settings, perc_missing,
                                                           num_unique_values))
        else:
            data_variables["bin_as"].append(bin_as_categorical(settings, perc_missing,
                                                               num_unique_values))

    data_variables["bin_as_panel"] = data_variables["bin_as"].copy()
    ret = pd.DataFrame.from_dict(data_variables)
    return ret.sort_values(by=["variable"]).reset_index(drop=True)


def calculate_bin_variables(df_input: pd.DataFrame, settings: BinningSettings, target_label: str,
                            df_data_vars: pd.DataFrame) -> pd.DataFrame:
    """Generate dataframe with binning criteria via the weight of evidence (WOE) approach."""
    if settings is None:
        settings = BinningSettings()

    if target_label is None:
        return pd.DataFrame()

    df_bin_criteria = \
        df_data_vars[df_data_vars["bin_as"] != BinAsMode.NONE][["variable", "bin_as"]]

    var_series, bin_as_series = df_bin_criteria["variable"], df_bin_criteria["bin_as"]
    woe = WeightOfEvidence()
    ret = woe.calculate_woe_iv_from_df(
          df=setup_binning_dataset(target_label, var_series.tolist(), bin_as_series.tolist(),
                                   df_input),
          var_list=var_series.tolist(),
          target_label=target_label,
          n_max_num=settings.numeric.max_num_bins,
          n_max_cat=settings.categorical.max_num_bins,
          round_limit=settings.numeric.decimal_place,
          lower_limit_bounded=setup_binning_lower_bound(var_series, bin_as_series),
          upper_limit_bounded=setup_binning_upper_bound(var_series, bin_as_series),
    )
    col_list = ["variable_parent"]
    col_list += ret.columns.tolist()
    ret["variable_parent"] = ret["variable"]
    ret["variable"] = "woe_" + ret["variable"]
    ret = ret[col_list]

    return ret
