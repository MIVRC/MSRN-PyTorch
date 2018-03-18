# Function：data.py
# Author：MIVRC
# Time：2018.2.1

import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)
        # input|hr_x2|hr_x4|hr_x8|bicubic_x2|bicubic_x4|bicubic_x8
        # 分别代表输入的LR、2倍放大的HR、4倍放大的HR、8倍放大的HR、2倍bicubic放大的SR、4倍bicubic放大的SR、8倍bicubic放大的SR
        # input:32*32; 2x:64*64; 4x:128*128; 8x:256*256
        self.data = dataset.get("data")
        self.label = dataset.get("label")

    def __getitem__(self, index):
        # 将数从numpy转为torch
        return torch.from_numpy(self.data[index, :, :, :]).float(), \
               torch.from_numpy(self.label[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]
