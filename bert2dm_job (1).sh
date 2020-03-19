#!/usr/bin/env bash

# Set job requirements
#SBATCH --job-name=bert2dm
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH --mem=90G
##SBATCH -c 1
##SBATCH -p gpu_shared
##SBATCH --cpus-per-task=1
##SBATCH --partition=gpu_shared
##SBATCH --gres=gpu:1

# Load all necessary modules
echo "Loading modules..."
# ? module load eb
module load 2019
module load Python/3.7.5-foss-2018b

# Load the python environment
source py375/bin/activate

# Copy the input data to scratch space
echo "Copying input data..."
cp -r ${HOME}/corpora/ ${TMPDIR}/corpora

# Create an output directory
echo "Creating an output directory: ${TMPDIR}/output/"
OUTPUT_DIR="${TMPDIR}/output/"
mkdir ${OUTPUT_DIR}

start=`date +%s`
# Run Python script
echo "Starting script..."
python -u ${HOME}/src/bert2dm.py \
	--corpus_path ${TMPDIR}/corpora/wikitext-2/wiki.train.tokens \
	--output_path ${TMPDIR}/output/bert2dm-768-wiki2.model \
	--dim 300 \
	--extract_layers 1 \
	--combine_layers no \
	--reduce_dim False
        &> ${TMPDIR}/output/log
end=`date +%s`
runtime=$((end-start))
echo "Runtime with unspecified cores was $runtime"

# --lexicon_file ${TMPDIR}/data/en-de.txt \
# --output_dir ${TMPDIR}/output --output_dim 250 --min_count 5

# Copy output from scratch
echo "Retrieving output"
cp -r ${TMPDIR}/output ${HOME}/output/${SLURM_JOBID}
