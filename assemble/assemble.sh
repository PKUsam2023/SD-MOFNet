#PBS -N Assemble
#PBS -l nodes=1:ppn=96
#PBS -l walltime=1000:00:00
#PBS -q nature

source .bashrc
conda activate mofdiff
cd ${raw_path}/SD-MOFNet-main/assemble

python mofdiff/scripts/assemble.py --input ${raw_path}/SD-MOFNet-main/data/diffusion_data/gen_csp_test_num_eval_100_processed.pt