from pathlib import Path





class DatasetTable:
    """Custom table used to display CSV and model data to end-user."""

    def __init__(self):
        super().__init__()

        self.setAcceptDrops(True)
        self.setDragEnabled(True)

