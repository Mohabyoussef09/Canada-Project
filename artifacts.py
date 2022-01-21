from analyze_artifacts import AnalyzeArtifacts
from backtest_artifacts import BacktestArtifacts
from binning_artifacts import BinningArtifacts
from train_artifacts import TrainArtifacts

class Artifacts:

    def __init__(self):
        super().__init__()
        self.analyze_artifacts = AnalyzeArtifacts()
        self.backtest_artifacts = BacktestArtifacts()
        self.binning_artifacts = BinningArtifacts()
        self.train_artifacts = TrainArtifacts()


    def reset(self):
        self.analyze_artifacts.reset()
        self.backtest_artifacts.reset()
        self.binning_artifacts.reset()
        self.train_artifacts.reset()
        self.reset_signal.emit()

    def update(self):
        self.changed_signal.emit()

    def serialize(self):
        return {"analyze": self.analyze_artifacts.serialize(),
                "binning": self.binning_artifacts.serialize(),
                "train": self.train_artifacts.serialize(),
                "backtest": self.backtest_artifacts.serialize(),
                }

    def deserialize(self, artifacts_serialized):
        # Note: Backtest must be done after training
        self.analyze_artifacts.deserialize(artifacts_serialized["analyze"])
        self.binning_artifacts.deserialize(artifacts_serialized["binning"])
        self.train_artifacts.deserialize(artifacts_serialized["train"])
        self.backtest_artifacts.deserialize(artifacts_serialized["backtest"])
