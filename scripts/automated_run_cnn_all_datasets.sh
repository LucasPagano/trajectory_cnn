#!/usr/bin/env bash


for item in eth hotel univ zara1 zara2;
do
  echo RUNNING automated_run_traj_cnn_multi.sh on $item
 /bin/bash automated_run_traj_cnn_multi.sh $item 0
done
