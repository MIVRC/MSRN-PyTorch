import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)

        self.data = dataset.get("data")
        self.label = dataset.get("label")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :]).float(), \
               torch.from_numpy(self.label[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]
