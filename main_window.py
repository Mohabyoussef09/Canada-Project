from dataclasses import dataclass
from pathlib import Path
import json
import threading
import pandas as pd

from artifacts import Artifacts
#from analyze_window import AnalyzeWindow
#from backtest_window import BacktestWindow
#from binning_window import BinningWindow
from dataset import Dataset
#from dataset.importer import DatasetImporter, DatasetImportResult
from dataset_table import DatasetTable
from dataset_tablemodel import DatasetTableModel
#from report_window import ReportWindow
from settings import Settings
#from toolbar import ToolBar
from train_settings import ModelType
#from train_window import TrainWindow



@dataclass
class MainWindowParameters:
    # Prompt the user to save their work when quitting the application
    prompt_save_on_quit: bool = True


class MainWindow:
    def __init__(self, params=MainWindowParameters()):
        super(MainWindow, self).__init__()

        self.settings = Settings()
        self.artifacts = Artifacts()
        self._file_dialog = None
        self.ds = Dataset()
        self.ds.data_changed.connect(self.data_changed)
        self.ds.target_changed.connect(self.on_target_changed)
        self.ds.binned_features_changed.connect(self.on_binned_features_changed)
        self.ds.excluded_vars_changed.connect(self.on_excluded_vars_changed)
        self.ds.probability_column_changed.connect(self.on_probability_column_changed)

        #self.ds_importer = DatasetImporter()
        self.ds_importer.import_started.connect(self.on_import_started)
        self.ds_importer.import_completed.connect(self.on_import_completed)

        self.ds_lock = threading.RLock()
        self.ds_imported = False # True when user has imported the a dataset
        self.ds_import_thread = None

        self.model_file = None
        self.params = params

    def construct_model_file(self):
        """Construct model file from artifacts."""
        mdl = {
            "dataset": self.ds.serialize(),
            "artifacts": self.artifacts.serialize(),
            "binning_window": self.binning_window.serialize(),
            "analyze_window": self.analyze_window.serialize(),
        }
        return mdl

    def open_action(self):
        """'Open' option for .model file."""

    def save_action(self):
        """'Save' option for .model file."""
        if self.model_file is not None:
            mdl = self.construct_model_file()
            with open(self.model_file, "w+") as f:
                json.dump(mdl, f)
        else:
            self.save_as_action()

    def save_as_action(self):
        """'Save As' option for .model file."""
        model_file= "filename"
        if model_file:
            self.model_file = model_file
            mdl = self.construct_model_file()
            with open(self.model_file, "w+") as f:
                json.dump(mdl, f)


    def on_load_training_dataset(self):
            probability_label = self.artifacts.train_artifacts.probability_label
            probability_series = self.artifacts.train_artifacts.train_df[
                probability_label
            ]
            df = self.artifacts.train_artifacts.train_df.drop(columns=probability_label)
            self.ds.load_df(probability_series, df)

    def closeEvent(self, event):
        # Note: This function is visited twice when an exit is made. We make sure that the exit
        # is permanent the second time. First time, event is Boolean; second time it is QCloseEvent
        if isinstance(event, bool):
            self.exit_pressed = True
        else:
            if self.exit_pressed:
                event.accept()
                return
            self.exit_pressed = False

        if not self.params.prompt_save_on_quit:
            return self.close()

