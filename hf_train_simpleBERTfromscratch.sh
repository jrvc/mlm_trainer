#!/bin/bash

#SBATCH --job-name=simBERT-tr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2001194
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=32G  
#SBATCH  -o log_%j
#SBATCH  -e log_%j

# USE:
#       cd /scratch/project_2001194/jrvc/LM-MTrepr/mt/logs/lm
#       sbatch /scratch/project_2001194/jrvc/LM-MTrepr/mt/scripts/hf_train_simpleBERTfromscratch.sh <src> <tgt> <nlayers>

module purge
deactivate
module load pytorch/1.9

currentdir=`pwd`

scriptpath=./scripts/
srclang=${1:-'en'}
tgtlang=${2:-'de'}
nlayers=${3:-'12'}

cd $scriptpath

nvidia-smi
echo "#### TRAINING LM WITH $nlayers LAYERS "
echo "     DATA FORM: ./data/corpora/${srclang}-${tgtlang}/ ####"

python $scriptpath/tok-and-trainer_simple_bert-from-scratch.py  \
                    --batch_size     32  \
                    --max_length    256  \
                    --train_epochs   50  \
                    --lca_steps      25  \
                    --warmup_steps 8000  \
                    --train_data_file ./data/corpora/${srclang}-${tgtlang}/  \
                    --n_hiddenlayers ${nlayers} \
                    --output_dir ./simpleBERTfromScratch/  \
                    --use_cuda

cd $currentdir
mv $currentdir/log_${SLURM_JOBID} /scratch/project_2001194/jrvc/LM-MTrepr/mt/logs/lm/train_simBERT_${srclang}-${tgtlang}_${SLURM_JOBID}.log


