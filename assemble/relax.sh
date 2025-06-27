#PBS -N relax_two
#PBS -l nodes=1:ppn=96
#PBS -l walltime=100:00:00
#PBS -q nature

source .bashrc
conda activate mofdiff
cd /share/home/fengb/Projects/MOFDiff_assembel

python mofdiff/scripts/uff_relax.py --input_folder /share/home/fengb/Projects/A/create_Result_data/plot/generation_result/RELAX/cif --cif_output_folder /share/home/fengb/Projects/A/create_Result_data/plot/generation_result/RELAX/relax --mof_id_folder /share/home/fengb/Projects/A/create_Result_data/plot/generation_result/RELAX/mofid > /share/home/fengb/Projects/MOFDiff_assembel/relax.txt