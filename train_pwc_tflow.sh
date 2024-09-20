#!/usr/bin/env bash

set -x
set -e

DESC=${@:1}

set -o pipefail

unset DISPLAY

ARGS=$(cat $1)
OUTPUT_BASE=$(echo $1 | sed -e "s/configs/exps\/$2/g" | sed -e "s/.args$//g")

mkdir -p $OUTPUT_BASE

for RUN in $(seq 1000); do
  ls $OUTPUT_BASE | grep -q run$RUN && continue
  if [ $RUN -lt 10 ]; then
    RUN=0$RUN
  fi
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

# run backup
echo "Backing up to log dir: $OUTPUT_DIR"
cp -r data_utils PWCNet dist_utils.py loss.py main_pwc_tflow.py $OUTPUT_DIR
echo " ...Done"


# log git status
echo "Logging git status"
git status > $OUTPUT_DIR/git_status
git rev-parse HEAD > $OUTPUT_DIR/git_tag
git diff > $OUTPUT_DIR/git_diff

pushd $OUTPUT_DIR
echo $DESC > desc
echo $2 > meme
echo " ...Done"

source /mnt/data_ssd/miniforge3/etc/profile.d/conda.sh
conda activate e2e_dy
# python -m torch.distributed.launch --nproc_per_node=4 main_pwc_tflow.py $ARGS |& tee output.log
export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_addr='127.0.0.1' --master_port=6735 main_pwc_tflow.py $ARGS |& tee output.log