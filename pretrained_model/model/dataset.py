import os
import torch

from os.path import join
from torch.utils.data import Dataset

# cif_lattice[64, 3, 3], cif_atom_positions[64, 50, 3], xrd[64, 5250], metal[64, 65], linker[64, Data(x=[a, 37], edge_index=[2, b], edge_attr=[b, 10])]


class CLIPDataset(Dataset):
    def __init__(self, mode, cfg):
        """
        参数：
        - mode (str): 数据集模式，可选值为 'train', 'test', 'valid'。
        """
        super().__init__()
        self.cfg = cfg
        self.data_path = self._get_data_path(mode)
        self.data = torch.load(self.data_path)


    def __len__(self):
        """返回数据集的大小。"""
        return len(self.data)


    def __getitem__(self, index):
        """
        根据索引加载数据，并返回 CIF 和 XRD 数据以及文件名标识符。

        参数：
        - index (int): 数据的索引。

        返回：
        - cif (torch.Tensor): 归一化后的 CIF 数据张量。
        - xrd (torch.Tensor): 归一化后的 XRD 数据张量。
        - metal (torch.Tensor): 归一化后的 Metal 数据张量。
        - tokenizer (str): 文件名标识符。
        """

        prefix = self.data[index]["prefix"]
        cif_lattice = self.data[index]["cif_lattice"]
        cif_atom_positions = self.data[index]["cif_atom_positions"]
        xrd = self.data[index]["xrd"]
        metal = self.data[index]["metal"]
        linker = self.data[index]["linker"]

        return cif_lattice, cif_atom_positions, xrd, metal, linker, prefix


    def _get_data_path(self, mode):
        if mode in ['train', 'test', 'valid']:
            # return join(self.cfg.data_path, mode + ".pt")
            return self.cfg.feature_extract_data
        else:
            raise ValueError(f"Invalid Mode: {mode}")
