import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from model.dataset import CLIPDataset

def collate_fn(batch):
    """
    自定义 collate_fn 函数，将批次数据打包为张量或 Batch，
    张量用于普通处理，Batch用于图数据处理。
    
    参数：
    - batch: 一个列表，每个元素是 CLIPDataset 返回的数据元组。
    
    返回：
    - 一个包含以下元素的元组：
        - cif_lattice_batch: torch.Tensor
        - cif_atom_positions_batch: torch.Tensor
        - xrd_batch: torch.Tensor
        - metal_batch: torch.Tensor
        - linker_batch: torch_geometric.data.Batch
        - tokenizer_batch: list
    """
    cif_lattice_batch = torch.stack([item[0] for item in batch])
    cif_atom_positions_batch = torch.stack([item[1] for item in batch])
    xrd_batch = torch.stack([item[2] for item in batch])
    metal_batch = torch.stack([item[3] for item in batch])
    linker_batch = Batch.from_data_list([item[4] for item in batch])  # 打包成 PyG Batch
    tokenizer_batch = [item[5] for item in batch]  # 保持列表格式

    return cif_lattice_batch,cif_atom_positions_batch, xrd_batch, metal_batch, linker_batch, tokenizer_batch


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """
        参数:
            batch_size: 批次大小
            num_workers: DataLoader 使用的多进程数量
            data_path: 数据根目录，包含对应的 .pt 文件
        """
        super().__init__()
        self.cfg = cfg

        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.data_path = self.cfg.data_path

    def setup(self, stage=None):
        """
        根据不同阶段加载对应数据集。
        如果 stage 为 'fit' 或 None，则加载训练集和验证集。
        如果 stage 为 'test' 或 None，则加载测试集。
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = CLIPDataset(mode="train", cfg = self.cfg)
            self.valid_dataset = CLIPDataset(mode="valid", cfg = self.cfg)
        if stage == 'test' or stage is None:
            self.test_dataset = CLIPDataset(mode="test", cfg = self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory = True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )
    