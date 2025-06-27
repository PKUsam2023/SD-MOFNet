#PBS -N pp_MOF_50
#PBS -l nodes=1:ppn=96
#PBS -l walltime=1000:00:00
#PBS -q nature


source .bashrc
conda activate mofdiff
cd /share/home/fengb/Projects/MOFDiff_assembel

python mofdiff/preprocessing/extract_mofid.py --df_path /share/home/fengb/Projects/A/create_Result_data/plot/EXP/MOF_ID/mof_id.csv --cif_path /share/home/fengb/Projects/A/create_Result_data/plot/EXP/MOF_ID/cif --save_path /share/home/fengb/Projects/A/create_Result_data/plot/EXP/MOF_ID/mof_id > /share/home/fengb/Projects/MOFDiff_assembel/a_EXP.o945988
# python preprocess.py --df_path /share/home/fengb/Projects/MOFDiff2/MOFDiff-main/assemble_test/CGmof50.csv --mofid_path /share/home/fengb/Data/Database/MOFtopo_Data/MOF50/mofid_50 --save_path /share/home/fengb/Data/Database/MOFtopo_Data/MOF50/graph > /share/home/fengb/Projects/MOFDiff2/MOFDiff-main/t2.txt
# python mofdiff/preprocessing/save_to_lmdb.py --graph_path /share/home/fengb/Data/Database/MOFtopo_Data/CG_xrd_Data/get_CG_xrd/CG_ligand_80978 --save_path /share/home/fengb/Projects/A/create_Generation_data/create_total_bbs_data > /share/home/fengb/Projects/MOFDiff2/MOFDiff-main/t3.txt
