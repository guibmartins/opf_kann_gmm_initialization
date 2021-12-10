#!/bin/bash
cd ../
repeat=$1
max_k=$2
dataset=$3
seed=1902
./venv/bin/python3.8 experiments_final.py $repeat $dataset $max_k $seed

