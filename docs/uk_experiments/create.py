from datasets.frs import (
    RawFRS_2018_19,
    RawFRS_2019_20,
    RawFRS_2020_21,
    FRS_2018_19,
    FRS_2019_20,
    FRS_2020_21,
    FRS_2018_21,
    FRS_2020_OUT_22,
    FRS_2018_21_OUT_22,
    CalibratedFRS_2020_21_22,
    CalibratedFRS_2018_21_22,
)
from datasets.spi import RawSPI_2019_20, SPI_2019_20

from pathlib import Path
import yaml
import logging


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with open(Path(__file__).parent / "local_setup.yaml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader).get("tab_folders")

    logging.info("Generating Raw FRS 2018-19")
    raw_frs_18 = RawFRS_2018_19()
    if not raw_frs_18.exists:
        raw_frs_18.generate(config.get("frs_2018_19"))

    logging.info("Generating Raw FRS 2019-20")
    raw_frs_19 = RawFRS_2019_20()
    if not raw_frs_19.exists:
        raw_frs_19.generate(config.get("frs_2019_20"))

    logging.info("Generating Raw FRS 2020-21")
    raw_frs_20 = RawFRS_2020_21()
    if not raw_frs_20.exists:
        raw_frs_20.generate(config.get("frs_2020_21"))

    logging.info("Generating FRS 2018-19")
    frs_18 = FRS_2018_19()
    if not frs_18.exists:
        frs_18.generate()

    logging.info("Generating FRS 2019-20")
    frs_19 = FRS_2019_20()
    if not frs_19.exists:
        frs_19.generate()

    logging.info("Generating FRS 2020-21")
    frs_20 = FRS_2020_21()
    if not frs_20.exists:
        frs_20.generate()

    logging.info("Generating FRS 2018-21")
    frs_21 = FRS_2018_21()
    if not frs_21.exists:
        frs_21.generate()

    logging.info("Generating Raw SPI 2019-20")
    raw_spi_19 = RawSPI_2019_20()
    if not raw_spi_19.exists:
        raw_spi_19.generate(config.get("spi_2019_20"))

    logging.info("Generating SPI 2019-20")
    spi_19 = SPI_2019_20()
    if not spi_19.exists:
        spi_19.generate()

    logging.info("Generating FRS 2020 OUT 22")
    frs_2020_out_22 = FRS_2020_OUT_22()
    if not frs_2020_out_22.exists:
        frs_2020_out_22.generate()

    logging.info("Generating FRS 2018 OUT 22")
    frs_2018_out_22 = FRS_2018_21_OUT_22()
    if not frs_2018_out_22.exists:
        frs_2018_out_22.generate()

    logging.info("Generating Calibrated FRS 2020-21-22")
    calibrated_frs_2020_21_22 = CalibratedFRS_2020_21_22()
    if not calibrated_frs_2020_21_22.exists:
        calibrated_frs_2020_21_22.generate()

    logging.info("Generating Calibrated FRS 2018-21-22")
    calibrated_frs_2018_21_22 = CalibratedFRS_2018_21_22()
    if not calibrated_frs_2018_21_22.exists or True:
        calibrated_frs_2018_21_22.generate()

