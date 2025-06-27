#PBS -N cal_embedding
#PBS -l nodes=g04:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source activate XXX
cd ${raw_path}/SD-MOFNet-main/pretrained_model

# Execute the Python script with arguments
python -u clip_cal_feature_embedding.py > clip_cal_feature_embedding.out