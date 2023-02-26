from datasets.frs import (
    SPIEnhancedFRS_2022,
    CalibratedFRS,
)
from pathlib import Path

CalibratedFRS.from_dataset(
    SPIEnhancedFRS_2022,
    force_generate=True,
    log_folder=Path(__file__).parent / "logs",
    verbose=True,
)
