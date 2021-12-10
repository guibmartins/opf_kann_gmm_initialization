#!/bin/bash
cd ../
repeat=$1
max_k=$2
seed=190220160
for dataset in frogs google_reviews ml1m; do
    ./venv/bin/python3.8 experiments_final.py $repeat $dataset $max_k $seed
done

