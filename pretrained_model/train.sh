#PBS -N train
#PBS -l nodes=g00:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source /share2/fengbin/anaconda3/bin/activate CLIP
cd /share2/fengbin/A/CLIPtopo_A

python -u clip_train.py > /share2/fengbin/A/CLIPtopo_A/clip_train.out
