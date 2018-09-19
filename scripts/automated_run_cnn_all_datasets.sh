#!/usr/bin/env bash


for item in eth hotel univ zara1 zara2;
do
  echo RUNNING automated_run_traj_cnn.sh on $item
 /bin/bash automated_run_traj_cnn.sh $item 0
done

echo EVALUATING on all datasets
python3 evaluate_cnn_all_datasets.py
