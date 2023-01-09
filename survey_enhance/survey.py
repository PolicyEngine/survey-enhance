"""
This module contains the main definition for the in-memory representation of survey datasets.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict

class Survey:
    """
    A `Survey` instance is an in-memory representation of a weighted survey.
    """

    name: str = None
    """A Python-safe name for the survey."""

    label: str = None
    """The label of the survey."""

    data_file: Path = None
    """The folder where the survey data is stored."""

    public_data_url: str = None
    """The URL where the public data is stored, if it is publicly available."""

    def __init__(self):
        if self.data_file is None:
            self.data_file = Path(__file__).parent / "data" / f"{self.name}.pkl"
        if not self.exists:
            self.generate()

    def generate(self):
        """Generate the survey data."""
        raise NotImplementedError(
            f"Survey {self.name} does not have a `generate` method."
        )

    def load(self, table_name: str) -> pd.DataFrame:
        """Load a table from disk."""
        with open(self.data_file, "rb") as f:
            tables: Dict[str, pd.DataFrame] = pickle.load(f)
        return tables[table_name]
        

    def save(self, tables: Dict[str, pd.DataFrame]):
        with open(self.data_file, "wb") as f:
            pickle.dump(tables, f)
    
    @property
    def exists(self) -> bool:
        return self.data_file.exists()
    
    def __repr__(self) -> str:
        return f"Survey({self.name!r}, {self.label!r})"
    
