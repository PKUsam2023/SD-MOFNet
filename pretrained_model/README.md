# Create your own test dataset

If you want to use this model to analyze your own experimental materials, your data must be converted into specific formats:
(All raw data should be placed under the ./data/feature_extract_data directory.)

## Extract Feature Embedding

Use the previously generated _dataset.pt_ file to compute feature embeddings.

Edit the project path in ./SD-MOFNet-main/pretrained_model/conf/config.yaml before running the following command:
```
python python -u clip_cal_feature_embedding.py > clip_cal_feature_embedding.out
```

## Process Feature Embedding

Use the output _dataset_emb.pt_ from the previous step and combine it with corresponding metal node and organic linker information.

Replace ${raw_path} with the path to your project directory. The BBS database valid_bbs_space_408626.pt can be downloaded separately.
```
python python -u script/process_feature_embedding.py --pt_file ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb.pt --smi_csv ${raw_path}/SD-MOFNet-main/data/original_data/original_data.csv --bbs_space ${raw_path}/SD-MOFNet-main/data/database/valid_bbs_space_408626.pt --output ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed.pt
```

## Create diffusion data

Use the processed _dataset_emb_processed.pt_ to calculate the number and ratio of atoms required for structure generation.

### Single-linker type

Replace ${raw_path} with your project directory:
```
python python -u cal_num_type_to_diffusion.py --input_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed.pt --output_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed_diff.pt
```

### Dual-linker type

Replace ${raw_path} with your project directory:
```
python python -u cal_num_type_to_diffusion.py --input_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed.pt --output_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed_diff.pt
```