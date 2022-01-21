from analyze_settings import AnalyzeSettings
from binning_settings import BinningSettings
from train_settings import TrainSettings


class Settings:
    def __init__(self):
        self.analyze_settings = AnalyzeSettings()
        self.binning_settings = BinningSettings()
        self.train_settings = TrainSettings()
