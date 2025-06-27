import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra

from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import seed_everything
from model.model import CLIPModule
from model.datamodel import CLIPDataModule

flagn = 1


def save_similarity_results(path, similarity_matrix, xrd_tokens, cif_tokens):
    """
    保存相似度矩阵和对应标签到 .npz 文件

    参数：
    - path: 输出文件路径（.npz）
    - similarity_matrix: numpy array, shape = [num_xrd, num_cif]
    - xrd_tokens: list[str], 行标签
    - cif_tokens: list[str], 列标签
    """
    np.savez_compressed(path,
                        similarity_matrix=similarity_matrix,
                        xrd_tokens=np.array(xrd_tokens),
                        cif_tokens=np.array(cif_tokens))
    print(f"相似度数据保存成功：{path}")


def find_matches(cfg, test_results, n=10):
    correct = 0
    unique_cif_embeddings = []  # 用于存储不重复 token 对应的 cif_embedding
    cif_filenames = []
    dot_similaritys = []
    xrd_tokens = []
    seen_tokens = set()
    for token, cif_emb, feature_emb in test_results:
        if token in seen_tokens:
            continue
        else:
            seen_tokens.add(token)
            cif_filenames.append(token)
            unique_cif_embeddings.append(cif_emb)

    unique_cif_embeddings_n = F.normalize(torch.stack(unique_cif_embeddings), p=2, dim=-1)

    for token, cif_emb, feature_emb in test_results:
        feature_embeddings_n = F.normalize(feature_emb, p=2, dim=-1)
        dot_similarity = feature_embeddings_n @ unique_cif_embeddings_n.T
        dot_similaritys.append(dot_similarity.cpu().detach().numpy())
        xrd_tokens.append(token)
        #Record
        global flagn
        if flagn == 1:
            with open(cfg.accuracy_file_path, "a+") as f:
                f.write(f"dot_similarity = {dot_similarity}\nmax dot_similarity = {torch.max(dot_similarity)}\nmin dot_similarity = {torch.min(dot_similarity)}\n")
            flagn += 1

        # Top N match
        values, indices = torch.topk(dot_similarity.squeeze(0), n)
        matches = [cif_filenames[idx] for idx in indices]
        flag = 0
        for match in matches:
            if token == match:
                flag = 1
                if flagn == 2 and 3:
                    with open(cfg.accuracy_file_path, "a+") as f:
                        f.write(f"cif = {str(token)}\nmatch = {str(matches)}\naccuracy = {str(values)}\n")
                    flagn += 1
                break
        if flag:
            correct += 1
        
    accuracy = correct / len(test_results)
    similarity_matrix = np.array(dot_similaritys)  # shape: [num_xrd, num_cif]
    # save_similarity_results(os.path.join(cfg.similarity_data_path, 'sim_mat.npz'), similarity_matrix, xrd_tokens, cif_filenames)

    return accuracy, len(test_results)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    start = time.time()
    seed = 512
    seed_everything(seed)

    # 创建模型，并移动到配置的设备上
    model = CLIPModule(cfg).to(cfg.device)
    
    # Load checkpoint (if exist)
    ckpts = list(Path(cfg.save_model_path).glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([
            int(ckpt.parts[-1].split('.')[0].split('_')[-1].split('=')[1])
            for ckpt in ckpts
        ])
        # 根据 epoch 数值排序，取最新（最大）的检查点文件
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        print("选定的ckpt文件名：", ckpt)
    else:
        ckpt = None

    # 构造测试数据模块，加载测试 DataLoader
    model = CLIPModule.load_from_checkpoint(ckpt)
    data = CLIPDataModule(cfg)
    data.setup(stage="test")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        devices=1 if torch.cuda.is_available() else None,
        accelerator='gpu',
        deterministic=True,
        progress_bar_refresh_rate=20
    )

    # 测试推理：调用 CLIP 函数获取测试结果
    trainer.test(model, data)

    test_results = model.all_test_results

    # 计算匹配准确率和相似度数据
    accuracy, data_len = find_matches(cfg, test_results, n=cfg.top_n)

    print(f"Test Accuracy: {accuracy} based on {data_len} samples")
    
    end = time.time()
    # 将准确率写入指定文件
    with open(cfg.accuracy_file_path, "a+") as f:
        f.write(f"accuracy = {accuracy}\n")
        f.write(f"all xrd = {data_len}\n")
        f.write(f"Test completed in {end - start:.2f} seconds\n")

if __name__ == '__main__':
    main()
