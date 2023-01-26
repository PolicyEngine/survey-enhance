from datasets.frs.raw_frs import RawFRS_2019_20

from pathlib import Path
import yaml
import logging


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open(Path(__file__).parent / "local_setup.yaml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader).get("tab_folders")

    logging.info("Generating Raw FRS 2018-19")
    raw_frs_18 = RawFRS_2019_20()
    if not raw_frs_18.exists:
        raw_frs_18.generate(config.get("frs_2019_20"))

        