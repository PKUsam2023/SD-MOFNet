import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pytorch_lightning as pl
import torch
import time
import numpy as np
import pandas as pd
import hydra
import re
from tqdm import tqdm
from copy import deepcopy
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import seed_everything
from model.model import CLIPModule
from model.datamodel import CLIPDataModule


def save_feature_emb(cfg, test_results):
    existing_data = torch.load(cfg.diff_data_path_ori)

    updated_data = []

    for item in existing_data:
        mof_id = item.get("mof_id", "")
        
        # Check if the mof_id matches any file in features_dict
        for token, feature_emb in test_results:
            
            # Extract the prefix from filename (remove '_数字' suffix)
            prefix = re.sub(r'_\d+$', '', token)
            
            if prefix == mof_id:
                # Create a copy of the original item
                new_item = deepcopy(item)
                
                # Modify the copied item with the new 'mof_id' and 'feature_embedding'
                new_item["mof_id"] = token  # Update 'mof_id' to filename
                new_item["feature_embedding"] = feature_emb  # Add feature_embedding
                
                # Append the updated item to the new data list
                updated_data.append(new_item)

    # Save the updated list of dictionaries to the new .pt file
    torch.save(updated_data, cfg.output_feature_embedding_data)
    print(f"Updated and saved new .pt file to {cfg.output_feature_embedding_data}.")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    seed = 512
    seed_everything(seed)

    # Create model
    model = CLIPModule(cfg).to(cfg.device)
    
    # Load checkpoint (if exist)
    ckpts = list(Path(cfg.save_model_path).glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([
            int(ckpt.parts[-1].split('.')[0].split('_')[-1].split('=')[1])
            for ckpt in ckpts
        ])
        # select checkpoint
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        print("选定的ckpt文件名：", ckpt)
    else:
        ckpt = None

    # load data
    model = CLIPModule.load_from_checkpoint(ckpt)
    data = CLIPDataModule(cfg)
    data.setup(stage="test")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        gpus=1 if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None and torch.cuda.is_available() else 0,
        accelerator='gpu',
        # distributed_backend='ddp',
        deterministic=True,
        progress_bar_refresh_rate=20
    )

    # get results
    trainer.test(model, data)
    test_results = model.all_test_results

    test_results_cpu = []
    for token, cif_emb, feature_emb in test_results:
        feature_emb = feature_emb.cpu().numpy()
        test_results_cpu.append((token, feature_emb))
    
    save_feature_emb(cfg, test_results_cpu)
    # torch.save(test_results_cpu, cfg.diff_data_emb_path)
    
    end = time.time()
    print(f"Test completed in {end - start:.2f} seconds\n")


if __name__ == '__main__':
    main()

