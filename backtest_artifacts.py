"""Retain model artifacts during the life of the application"""

import pandas as pd


class BacktestArtifacts:

    def __init__(self):
        super().__init__()
        self.kpis_df = None
        self.calibration_df = None
        self.psi_df = None
        self.ranking_df = None
        self.distribution_df = None

    def reset(self):
        self.kpis_df = None
        self.calibration_df = None
        self.psi_df = None
        self.ranking_df = None
        self.distribution_df = None
        self.reset_signal.emit()

    def update_all(self, kpis_df, calibration_df, psi_df, ranking_df, distribution_df):
        self.kpis_df = kpis_df.copy()
        self.calibration_df = calibration_df.copy()
        self.psi_df = psi_df.copy()
        self.ranking_df = ranking_df.copy()
        self.distribution_df = distribution_df.copy()
        self.update()

    def update(self):
        self.changed_signal.emit()

    @property
    def all_artifacts_available(self):
        if self.kpis_df is not None and \
                self.calibration_df is not None and \
                self.psi_df is not None and \
                self.ranking_df is not None and \
                self.distribution_df is not None:
            return True
        return False

    def serialize(self):
        artifacts_dict = {
            "all_available": False,
            "kpis": "",
            "calibration": "",
            "stability": "",
            "ranking": "",
            "distribution": "",
        }
        if self.all_artifacts_available:
            artifacts_dict["all_available"] = True

            artifacts_dict["kpis"] = self.kpis_df.to_json()

            artifacts_dict["calibration"] = self.calibration_df.to_json()

            artifacts_dict["stability"] = self.psi_df.to_json()

            artifacts_dict["ranking"] = self.ranking_df.to_json()

            artifacts_dict["distribution"] = self.distribution_df.to_json()

        return artifacts_dict

    def deserialize(self, artifacts_serialized):
        self.reset()

        if artifacts_serialized["all_available"]:
            self.kpis_df = pd.read_json(artifacts_serialized["kpis"])

            self.calibration_df = pd.read_json(artifacts_serialized["calibration"])

            self.psi_df = pd.read_json(artifacts_serialized["stability"])

            self.ranking_df = pd.read_json(artifacts_serialized["ranking"])

            self.distribution_df = pd.read_json(artifacts_serialized["distribution"])

            self.update()
