# Assemble Generation Data

Assemble the generated structure data into atomic structures.

## Acknowledgements and License

This project is adapted from the code developed by Microsoft Corporation and Massachusetts Institute of Technology.
[paper](https://arxiv.org/abs/2310.10732) | [data and pretained models](https://zenodo.org/uploads/10467288)
The original code is released under the MIT License, reproduced below:

    MIT License

    Copyright (c) Microsoft Corporation and Massachusetts Institute of Technology.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE

## Requirement

Install _mofdiff_:
```
pip install -e .
```

## Process

### Single-linker type process

Replace ${raw_path} with your project directory path:
```
python Single_linker_reorganize.py --input_pt ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100.pt --bbs_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed_diff.pt --output_pt ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed.pt
```

### Dual-linker type process

Use multiprocessing to improve efficiency.

Replace ${raw_path} with your project directory path:
```
python Dual_data_split.py --input_pt ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100.pt --bbs_pt ${raw_path}/SD-MOFNet-main/data/feature_extract_data/dataset_emb_processed_diff.pt --output_dir ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed
python Dual_ewald_summation.py --input_dir ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed --output_dir ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed_pro --workers 20
python Dual_data_merge.py --input_folder ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed_pro --output_file ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed.pt
```


## Assemble

```
python mofdiff/scripts/assemble.py --input ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed.pt
```

After completion, you will find the generated _.cif_ files in the _diffusion_data_ folder.

## Relax process

We use [MOFid](https://github.com/snurr-group/mofid) for analysis. To perform these steps, install MOFid following the instruction in the [MOFid repository](https://github.com/snurr-group/mofid/blob/master/compiling.md). The generative modeling and MOF simulation portions of this codebase do not depend on MOFid.

```
python mofdiff/scripts/uff_relax.py --input_folder ${raw_path}/SD-MOFNet-main/data/diffusion_data/cif --cif_output_folder ${raw_path}/SD-MOFNet-main/data/diffusion_data/relax --mof_id_folder ${raw_path}/SD-MOFNet-main/data/diffusion_data/mofid
```