"""@copyright Copyright © 2021, Modellicity Inc., All Rights Reserved."""

from modellicity.src.modellicity.data_source import DataSource

import config

import numpy as np
import pandas as pd

class Dataset:

    def __init__(self):
        super().__init__()
        self.ds = None

        self.initial_var_order = None
        self.selected_target = None
        self.valid_target_vars = None
        self.invalid_vars = None
        self.excluded_vars = None
        self.binned_feature_vars = None
        self.properties = None
        self.reset()

    @staticmethod
    def reset_dataset():
        return pd.DataFrame(index=range(1, config.DEFAULT_ROW_COUNT + 1),
                            columns=range(1, config.DEFAULT_COL_COUNT + 1))

    def reset(self):
        self.ds = DataSource()
        self.ds.data = self.reset_dataset()

        self.initial_var_order: dict[str, int] = {}
        self.selected_target = None
        self.valid_target_vars: list[int] = []
        self.invalid_vars: list[int] = list(range(config.DEFAULT_COL_COUNT))
        self.excluded_vars: list[int] = list(range(config.DEFAULT_COL_COUNT))
        self.binned_feature_vars: list[int] = []
        self.properties = {
            # True if DataFrame is empty or only contains "missing" values
            "is_empty": True
        }


    @property
    def data(self):
        return self.ds.data

    @property
    def is_empty(self):
        return self.properties.get("is_empty", True)

    @property
    def valid_targets(self) -> list[tuple[int, str]]:
        return [(idx, self.ds.data.columns[idx]) for idx in self.valid_target_vars]

    @property
    def probability_label(self):
        return "Probability"

    @property
    def target_label(self):
        if self.selected_target is not None:
            return list(self.ds.data.columns)[self.selected_target]

    @property
    def non_event_labels(self):
        non_events = (
            [self.probability_label]
            if self.target_label is None else [self.probability_label, self.target_label]
        )
        return [x for x in self.ds.data.columns if x not in non_events]

    @property
    def model_qualified_var_labels(self):
        if self.non_event_labels:
            return [
                x
                for x in self.non_event_labels
                if x not in self.ds.data.columns[self.invalid_vars]
            ]


    @property
    def binned_feature_var_labels(self):
        binned_feature_var_labels = []
        for i, col_name in enumerate(self.ds.data.columns):
            if i in self.binned_feature_vars:
                binned_feature_var_labels.append(col_name)
        return binned_feature_var_labels

    def load_df(self, probability_series: pd.Series, df: pd.DataFrame):
        self.ds.data = df.copy()
        self.ds.data.insert(0, self.probability_label, probability_series.to_numpy())
        self.update()

    def update(self):
        """Update internal data. Call after the datasource has changed."""
        self.selected_target = None
        self.initial_var_order: dict[str, int] = {}
        self.valid_target_vars = []
        self.binned_feature_vars = []

        for i, col_name in enumerate(self.ds.data.columns):
            self.initial_var_order[col_name] = i
            # We are assuming valid target columns to contain the values {0, 1} only and only
            # non-feature columns qualify
            if set(self.ds.data[col_name].unique()) == {0, 1}:
                self.valid_target_vars.append(i)

        self.update_is_empty()
        self.update_invalid_vars_list()


    def update_invalid_vars_list(self):
        """Flag non-model variables: categorical variables, variables with missing values."""
        categorical_list = self.ds.data.select_dtypes(include="object").columns.tolist()
        missing_list = [x for x in self.ds.data.columns.tolist() if x not in
                        [self.probability_label] + categorical_list +
                        self.ds.data.dropna(axis=1).columns.tolist()]
        self.invalid_vars = list(np.sort(self.ds.data.columns.get_indexer(categorical_list
                                                                          + missing_list)))
        # TODO #421: Find a way to serialize int64 and avoid line below.
        self.invalid_vars = [int(x) for x in self.invalid_vars]

        excluded_vars_old = self.excluded_vars.copy()
        self.excluded_vars = self.invalid_vars.copy()


    def update_excluded_vars_list(self, user_excluded_vars: list[int]):
        excluded_vars_old = self.excluded_vars.copy()
        if user_excluded_vars is None:
            return
        if len(user_excluded_vars) == 0:
            self.excluded_vars = self.invalid_vars.copy()
        else:
            if (
                min(user_excluded_vars) < 0
                or max(user_excluded_vars) >= self.ds.data.shape[1]
            ):
                return
            self.excluded_vars = sorted(
                list(set(self.invalid_vars).union(set(user_excluded_vars)))
            )


    def update_is_empty(self):
        if self.ds.data.size == 0:
            self.properties["is_empty"] = True
        else:
            self.properties["is_empty"] = (self.ds.data.dropna(axis=1).shape[1] == 0)

    def update_probability_column(self, probs: np.ndarray):
        """Update the probability column with updated probability entries."""
        self.ds.data[self.probability_label] = probs


    def update_selected_target(self, target_label: str):
        self.reset_binned_feature_vars()
        initial_var_labels = list(self.initial_var_order.keys())

        if target_label not in initial_var_labels:
            self.selected_target = None
            if initial_var_labels:
                self.ds.data = self.ds.data.reindex(columns=initial_var_labels)
                self.update_invalid_vars_list()

            return

        selected_target_original_location = self.initial_var_order[target_label]

        # Case: selected target variable is located in its destination place
        if selected_target_original_location == 1:
            self.selected_target = 1
            self.ds.data = self.ds.data.reindex(columns=initial_var_labels)
        # Case: selected target variable is located after its destination place
        else:
            cols = [self.probability_label, target_label]
            for i in range(2, len(initial_var_labels)):
                if i <= selected_target_original_location:
                    cols.append(initial_var_labels[i - 1])
                else:
                    cols.append(initial_var_labels[i])
            self.selected_target = 1
            self.ds.data = self.ds.data.reindex(columns=cols)
        self.update_invalid_vars_list()

    '''
    def add_binned_feature_vars(self, df_binned_features: pd.DataFrame):
        # First, remove all existing feature columns
        self.reset_binned_feature_vars()
        if self.ds.data.shape[0] == df_binned_features.shape[0]:
            self.binned_feature_vars = []
            num_cols = self.ds.data.shape[1]
            for i, col_name in enumerate(df_binned_features):
                self.ds.data[col_name] = df_binned_features[col_name]
                self.binned_feature_vars.append(num_cols + i)
            self.update_invalid_vars_list()

        return self.ds.data
    '''

    def add_binned_feature_vars(self,df_original: pd.DataFrame ,df_binned_features: pd.DataFrame):
        # First, remove all existing feature columns
        #self.reset_binned_feature_vars()
        if df_original.shape[0] == df_binned_features.shape[0]:
            self.binned_feature_vars = []
            num_cols = df_original.shape[1]
            for i, col_name in enumerate(df_binned_features):
                df_original[col_name] = df_binned_features[col_name]
                self.binned_feature_vars.append(num_cols + i)
            self.update_invalid_vars_list()

        return df_original

    def reset_binned_feature_vars(self):
        if len(self.binned_feature_vars) > 0:
            self.ds.data.drop(columns=self.binned_feature_var_labels, inplace=True)
            self.binned_feature_vars = []
            self.update_invalid_vars_list()


    def serialize(self):
        return {
            "data": self.ds.data.to_json(),
            "selected_target": self.selected_target,
            "binned_feature_vars": self.binned_feature_vars,
            "excluded_vars": self.excluded_vars,
        }

    def deserialize(self, dataset_serialized_dict):
        # 1: Reset everything
        self.reset()

        # 2: Update non-feature data (and store feature data)
        df_all = pd.read_json(dataset_serialized_dict["data"])
        binned_feature_vars = dataset_serialized_dict["binned_feature_vars"]
        non_binned_feature_vars = [
            x for x in list(range(df_all.shape[1])) if x not in binned_feature_vars
        ]
        self.ds.data = df_all.iloc[:, non_binned_feature_vars].copy()
        self.update()

        # 3: Update selected target
        self.update_selected_target(dataset_serialized_dict["selected_target"])

        # 4: Update feature vars
        if len(binned_feature_vars) > 0:
            self.add_binned_feature_vars(df_all.iloc[:, binned_feature_vars])

        # 6: Update excluded vars
        user_excluded_vars = [
            x
            for x in dataset_serialized_dict["excluded_vars"]
            if x not in self.invalid_vars
        ]
        if len(user_excluded_vars) > 0:
            self.update_excluded_vars_list(user_excluded_vars)
