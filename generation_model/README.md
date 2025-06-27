# Generation Model

## Acknowledgements and License

This project is adapted from the original code developed by Rui Jiao (© 2023).
The original code is released under the MIT License, as shown below: 

> MIT License  
>  
> Copyright (c) 2023 Rui Jiao  
>  
> Permission is hereby granted, free of charge, to any person obtaining a copy  
> of this software and associated documentation files (the "Software"), to deal  
> in the Software without restriction, including without limitation the rights  
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
> copies of the Software, and to permit persons to whom the Software is  
> furnished to do so, subject to the following conditions:  
>  
> The above copyright notice and this permission notice shall be included in all  
> copies or substantial portions of the Software.  
>  
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
> SOFTWARE.

All modifications made in this repository are also released under the MIT License.
For details, see the LICENSE file in the root directory of this repository.

Rename the `.env.template` file into `.env` and specify the following variables.

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wabdb outputs
```

## Train
For details on the training process, please refer to the [MOFDiff](https://github.com/microsoft/MOFDiff)

## Multiple samples

For the CSP task

```
python scripts/evaluate.py --model_path ${raw_path}/SD-MOFNet-main/generation_model/output/hydra/singlerun/2025-04-03/CSP_CGmof50 --dataset CGmof_50 --num_evals 100 --label test > ${raw_path}/SD-MOFNet-main/generation_model/test.txt
```

