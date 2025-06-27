#PBS -N Generation
#PBS -l nodes=g00:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
source activate DIFFUSION
cd  ${raw_path}/SD-MOFNet-main/generation_model

python scripts/evaluate.py --model_path ${raw_path}/SD-MOFNet-main/generation_model/output/hydra/singlerun/2025-04-03/CSP_CGmof50 --dataset CGmof_50 --num_evals 100 --label ANY > ${raw_path}/SD-MOFNet-main/generation_model/test.txt