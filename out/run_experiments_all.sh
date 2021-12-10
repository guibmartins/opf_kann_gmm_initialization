#!/bin/bash
cd ../
repeat=$1
max_k=$2
seed=19022016
for dataset in bcw blood ccrf diabetic frogs google_reviews ml1m mm; do
    ./venv/bin/python3.8 experiments_final.py $repeat $dataset $max_k $seed
done

