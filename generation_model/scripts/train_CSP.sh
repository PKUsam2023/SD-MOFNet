#PBS -N Train
#PBS -l nodes=g00:ppn=8
#PBS -l walltime=100:00:00
#PBS -q ml

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
source activate DIFFUSION
cd ${raw_path}/SD-MOFNet-main/generation_model

python diffcsp/run.py data=CGmof_50 expname=CSP_CGmof50_layer_3 > ${raw_path}/SD-MOFNet-main/generation_model/train.txt
# tensorboard --logdir lightning_logs