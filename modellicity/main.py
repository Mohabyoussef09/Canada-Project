"""@copyright Copyright © 2019, Modellicity Inc., All Rights Reserved."""
import numpy as np
import pandas as pd
from modellicity.extended_pandas import ExtendedDataFrame


def main():
    """Implement main module for running quick tests."""
    df = pd.DataFrame(
        data={
            "X": [1, np.nan, 1, 1, 1],
            "Y": ["X", "X", "X", "X", "X"],
            "Z1": [1, 1, 1, 1, 1],
            "Z2": [1, 2, 3, 4, 5]
        }
    )

    df = ExtendedDataFrame(df)
    print(df.get_high_concentration_variables(0.6))


if __name__ == "__main__":
    main()
