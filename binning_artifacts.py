"""Retain binning artifacts during the life of the application"""
from binning_settings import BinAsMode, BinningSettings
import copy
import pandas as pd

class BinningArtifacts:
    def __init__(self):
        super().__init__()
        self.data_vars_df = None
        self.bin_vars_df = None
        self.binning_settings = None

    def __copy__(self):
        new_instance = BinningArtifacts()
        if self.data_vars_df is not None:
            new_instance.data_vars_df = self.data_vars_df.copy()
        if self.bin_vars_df is not None:
            new_instance.bin_vars_df = self.bin_vars_df.copy()
        if self.binning_settings is not None:
            new_instance.binning_settings = copy.copy(self.binning_settings)
        return new_instance

    def reset(self):
        self.data_vars_df = None
        self.bin_vars_df = None
        self.binning_settings = None
        self.reset_signal.emit()

    def update_all(self, data_vars_df, bin_vars_df, binning_settings):
        self.data_vars_df = data_vars_df.copy()
        if bin_vars_df is not None:
            self.bin_vars_df = bin_vars_df.copy()
        self.binning_settings = copy.copy(binning_settings)
        self.update()

    def update(self):
        self.changed_signal.emit()

    @property
    def all_artifacts_available(self):
        if self.data_vars_df is not None and \
                self.bin_vars_df is not None and \
                self.binning_settings is not None:
            return True
        return False

    @property
    def binned_vars_dict(self):
        if self.bin_vars_df is not None:
            temp1 = \
                self.bin_vars_df[["variable_parent", "variable",
                                  "iv"]].drop_duplicates().reset_index(drop=True)
            temp2a = temp1[["variable_parent", "iv"]].copy()
            temp2a.rename(columns={"variable_parent": "variable"}, inplace=True)
            temp2b = temp1[["variable", "iv"]]
            tempc = pd.concat([temp2a, temp2b], ignore_index=True).reset_index(drop=True).set_index(
                "variable")
            return tempc.to_dict()["iv"]
        return None

    def serialize(self):
        artifacts_dict = {
            "all_available": False,
            "data_vars": "",
            "bin_vars": "",
            "binning_settings": "",
        }
        # Note: unique to binning artifacts: it is possible for self.bin_vars_df to be None while
        # other objects are not
        if self.data_vars_df is not None:
            artifacts_dict["all_available"] = True

            data_vars_df = self.data_vars_df.copy()
            data_vars_df["bin_as_panel"] = \
                [str(x) for x in list(data_vars_df["bin_as_panel"])]
            data_vars_df["bin_as"] = \
                [str(x) for x in list(data_vars_df["bin_as"])]
            artifacts_dict["data_vars"] = data_vars_df.to_json()

            if self.bin_vars_df is not None:
                bin_vars_df = self.bin_vars_df.copy()
                artifacts_dict["bin_vars"] = bin_vars_df.to_json()

            artifacts_dict["binning_settings"] = self.binning_settings.serialize()

        return artifacts_dict

    def deserialize(self, artifacts_serialized):
        self.reset()

        if artifacts_serialized["all_available"]:
            self.data_vars_df = \
                pd.read_json(artifacts_serialized["data_vars"])
            self.data_vars_df["bin_as_panel"] = \
                [BinAsMode.from_str(x) for x in list(self.data_vars_df["bin_as_panel"])]
            self.data_vars_df["bin_as"] = \
                [BinAsMode.from_str(x) for x in list(self.data_vars_df["bin_as"])]

            if artifacts_serialized["bin_vars"] != "":
                self.bin_vars_df = \
                    pd.read_json(artifacts_serialized["bin_vars"])

            self.binning_settings = BinningSettings()
            self.binning_settings.deserialize(artifacts_serialized["binning_settings"])

            self.update()
