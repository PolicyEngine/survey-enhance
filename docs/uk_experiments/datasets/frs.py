from survey_enhance.survey import Survey
from pathlib import Path
import pandas as pd
import yaml

class RawFRS(Survey):
    """A `Survey` instance for the Family Resources Survey."""
    name = "frs"
    label = "Family Resources Survey"

    def generate(self, tab_folder: Path):
        """Generate the survey data from the original TAB files.

        Args:
            tab_folder (Path): The folder containing the original TAB files.
        """

        if isinstance(tab_folder, str):
            tab_folder = Path(tab_folder)
        
        # Load the data
        tables = {}
        for tab_file in tab_folder.glob("*.tab"):
            table_name = tab_file.stem
            tables[table_name] = pd.read_csv(tab_file, sep="\t")
        
        # Save the data
        self.save(tables)

if __name__ == "__main__":
    with open(Path(__file__).parent / "local_setup.yaml") as f:
        local_setup_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    
    raw_frs = RawFRS()
    raw_frs.generate(local_setup_yaml.get("raw_frs_tab_folder"))
