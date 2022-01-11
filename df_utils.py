import pandas as pd
from typing import List


data_types_map = {
    "int64": "Numeric",
    "float64": "Numeric",
    "object": "Categorical",
    "datetime64": "Datetime"
}


def get_all_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    """Obtain all variables of datetime format.

    Args:
        df: A dataframe with rows representing a variable and columns representing corresponding data.

    Returns:
        A list of column names that are of datetime format.
    """
    col_list = df.columns.tolist()
    is_datetime_format_list: List[bool] = [False] * len(col_list)
    non_numeric_columns = df.select_dtypes(exclude="number").columns
    for i, col in enumerate(col_list):
        if col in non_numeric_columns:
            try:
                series = pd.to_datetime(df[col])
                if not pd.isnull(series).all():
                    is_datetime_format_list[i] = True
            except ValueError:
                pass
    return pd.DataFrame({"variable": col_list, "is_datetime_format": is_datetime_format_list})


def get_percent_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Obtain percentage of missing values for each variable of dataframe.

    Args:
      df: A dataframe with rows representing a variable and columns representing corresponding data.

    Returns:
      A two-column dataframe where the first column corresponds to the variable name and the second corresponds to the
      respective percentage of missing values of that variable.
    """
    df_ret = pd.DataFrame({"variable": df.columns, "perc_missing": df.isnull().sum() / len(df)})
    df_ret = df_ret.sort_values(by=["perc_missing", "variable"], ascending=[False, True])
    return df_ret.set_index("variable", drop=True)


def get_variable_types(df: pd.DataFrame) -> pd.DataFrame:
    """Obtain information on specified data type of variable in dataframe.

    Args:
      df: A dataframe with rows representing a variable and columns representing corresponding data.

    Returns:
      A two-column dataframe where the first column corresponds to the variable name and the second corresponds to the
      variable type of data.
    """
    df_ret = pd.DataFrame({"variable": df.columns,
                           "type": df.dtypes.astype(str).map(data_types_map)}).sort_values(by=["variable"])
    return df_ret.set_index("variable", drop=True)


def get_num_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Obtain information on number of unique values in dataframe.

    Args:
      df: A dataframe with rows representing a variable and columns representing corresponding data.

    Returns:
      A two-column dataframe where the first column corresponds to the variable name and the second corresponds to the
      number of unique values of that variable.
    """
    df_ret = df.nunique(dropna=False).to_frame().reset_index()
    df_ret.columns = ["variable", "num_unique_values"]
    return df_ret.set_index("variable", drop=True)
