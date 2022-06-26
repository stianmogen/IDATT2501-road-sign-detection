import pandas as pd
from torch.utils.data import Dataset


class RoadSignDataset(Dataset):
    """Road Sign dataset."""

    def __init__(self, pickle_file, transform=None):
        data_frame = pd.read_pickle(pickle_file)
        self.features = data_frame["Features"].values
        self.labels = data_frame["Labels"].values
        self.size = len(data_frame)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.features[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]
