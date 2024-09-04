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
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

# run backup
echo "Backing up to log dir: $OUTPUT_DIR"
cp -r data_utils PWCNET dist_utils.py loss.py main_pwc.py $OUTPUT_DIR
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

conda activate e2e_dy
python -m torch.distributed.launch --nproc_per_node=4 main_pwc.py $ARGS |& tee output.log