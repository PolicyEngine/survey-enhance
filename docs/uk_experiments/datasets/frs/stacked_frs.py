from policyengine_core.data import Dataset
import numpy as np
from pathlib import Path


class StackedFRS(Dataset):
    sub_datasets = []
    weighting_factors = []

    @staticmethod
    def from_dataset(datasets, weight_factors, new_name, new_label):
        class StackedDatasetFromDataset(StackedFRS):
            sub_datasets = datasets
            weighting_factors = weight_factors
            name = new_name
            label = new_label
            data_format = datasets[0].data_format
            file_path = (
                Path(__file__).parent.parent.parent / "data" / f"{new_name}.h5"
            )

        return StackedDatasetFromDataset

    def generate(self):
        sub_datasets = [dataset() for dataset in self.sub_datasets]
        variable_names = sub_datasets[0].variables
        data = {}
        for variable in variable_names:
            if "_id" in variable:
                new_ids = []
                max_id = 0
                for dataset in sub_datasets:
                    new_ids.append(dataset.load(variable) + max_id)
                    max_id += dataset.load(variable).max()
                data[variable] = np.concatenate(new_ids)
            elif "_weight" in variable:
                data[variable] = np.concatenate(
                    [
                        dataset.load(variable) * weight
                        for dataset, weight in zip(
                            sub_datasets, self.weighting_factors
                        )
                    ]
                )
            else:
                data[variable] = np.concatenate(
                    [dataset.load(variable) for dataset in sub_datasets]
                )
        self.save_dataset(data)
