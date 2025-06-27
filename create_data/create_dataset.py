import os
import re
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join
import torch.multiprocessing as mp
from multiprocessing import Pool

mp.set_sharing_strategy('file_system')

def _load_xrd_data(file_path):
    """
    Load XRD data from a .pt file and normalize it (L2 norm).
    """
    xrd = torch.load(file_path)
    xrd = F.normalize(xrd, p=2, dim=0)
    return xrd

def _load_metal_data(metal_data, file_name):
    """
    Retrieve the pre-generated metal csv object from the dictionary.
    """
    metal_row = metal_data[metal_data['CG_cif_filename'] == file_name]
    metal_tensor = torch.tensor(eval(metal_row.iloc[0]['Metal_tensor']), dtype=torch.float32)
    metal = F.normalize(metal_tensor, p=2, dim=0)
    return metal

def _load_linker_data(linker_data, file_name):
    """
    Retrieve the pre-generated linker graph data object from the dictionary.
    """
    return linker_data[file_name]


def process_sample(prefix, xrd_dir, metal_df, linker_dict, xrd_ext='.pt'):
    """
    Process a single sample:
      - Load XRD data by prefix
      - Load metal features by prefix
      - Load linker graph by prefix
      Returns a dict containing all tensors and Data.
    """
    xrd_file = os.path.join(xrd_dir, f"{prefix}{xrd_ext}")
    xrd = _load_xrd_data(xrd_file)
    metal = _load_metal_data(metal_df, prefix)
    linker = _load_linker_data(linker_dict, prefix)

    return {
        'prefix': prefix,
        'xrd': xrd,
        'metal': metal,
        'linker': linker
    }

def prepare_dataset_to_pt(csv_path, output_pt, error_log_file):
    """
    Read prefixes from a CSV, then parallel-process each sample to build a dataset dict,
    and save it as a single .pt file.
    """
    data_path = os.path.dirname(csv_path) 
    xrd_path = join(data_path, "xrd")
    metal_path = join(data_path, "metal.csv")
    linker_path = join(data_path, "linker.pt")
    
    prefixes_df = pd.read_csv(csv_path)
    prefixes = prefixes_df['Materials_name'].astype(str).tolist()
    metal_data = pd.read_csv(metal_path)
    linker_data = torch.load(linker_path)
    
    results = []
    with open(error_log, 'w', encoding='utf-8') as err_f:
        for prefix in tqdm(prefixes, desc='Processing samples'):
            try:
                sample = process_sample(prefix, xrd_dir, metal_df, linker_dict)
                results.append(sample)
            except Exception as e:
                err_f.write(prefix + '\n')
    
    # Build indexed dict
    dataset = {i: sample for i, sample in enumerate(results)}
    torch.save(dataset, output_pt)
    print(f"Saved {len(results)} samples to {output_pt}.")

if __name__ == "__main__":
    # 指定数据路径和输出文件路径
    parser.add_argument('--input_csv', type=str, required=True,
                        help="CSV file with 'Materials_name' column for prefixes.")
    parser.add_argument('--output_pt', type=str, required=True,
                        help='Output path for the combined dataset .pt file.')
    parser.add_argument('--error_log_file', type=str, required=True,
                        help='Path to write prefixes that failed processing.')
    args = parser.parse_args()
    prepare_dataset_to_pt(input_csv, output_pt, error_log_file)
