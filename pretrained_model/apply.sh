#PBS -N apply
#PBS -l nodes=g06:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source /share2/fengbin/anaconda3/bin/activate CLIP
cd /share2/fengbin/A/CLIPtopo_A

# Execute the Python script with arguments
python -u clip_apply.py > /share2/fengbin/A/CLIPtopo_A/clip_apply.out
# clip_model.py -> test_step加去前缀代码