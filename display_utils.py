import pandas as pd


def fmt_apply(df: pd.DataFrame, key: str, num_digits: int, fmt: str) -> pd.DataFrame:
    return df.apply(lambda x: "{number:,.{digits}{fmt}}".format(number=x[key],
                                                                digits=num_digits,
                                                                fmt=fmt), axis=1)
