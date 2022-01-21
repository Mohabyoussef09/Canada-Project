from dataset import Dataset
from typing import Any
import config


class DatasetTableModel:
    def __init__(self, ds: Dataset):
        super().__init__()

        self.ds = ds
        self.ds.import_started.connect(self.beginResetModel)
        self.ds.import_completed.connect(self.endResetModel)
        self.ds.target_changed.connect(self.header_changed)
        self.ds.binned_features_changed.connect(self.header_changed)
        self.ds.probability_column_changed.connect(self.header_changed)

    # ---
    # QAbstractTableModel overrides
    # ---

