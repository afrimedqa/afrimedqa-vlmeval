#!/bin/bash

cat $0
set -x

export LMUData=~/afrimedqa-vlmeval/lmudata
export GPU=$(nvidia-smi --list-gpus | wc -l)
nvidia-smi

lang=en
question_type=MCQ
model=Gemma3-12B
img_path=images/afrimedqa

python run.py --data AfrimedQA --lang $lang --question-type $question_type --model $model  --img_path $img_path --work-dir results/


