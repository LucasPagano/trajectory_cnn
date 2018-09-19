#!/usr/bin/env bash
# use it from the scripts folder
# usage run_traj_cnn.sh dataset_name restore
export PYTHONPATH=`cd .. && pwd`

python3 train_cnn.py \
  --dataset_name $1 \
  --delim tab \
  --pred_len 12 \
  --encoder_h_dim_g 32 \
  --embedding_dim 32 \
  --num_layers 4 \
  --l2_loss_weight 1 \
  --dropout 0 \
  --batch_size 32 \
  --g_learning_rate 1e-3 \
  --print_every 200 \
  --num_epochs 50 \
  --checkpoint_name $1_50epoch \
  --checkpoint_every 50 \
  --output_dir save \
  --loader_num_workers 0 \
  --restore_from_checkpoint $2

echo "TESTING NOW epochs 50"
python3 evaluate_cnn.py --model_path save/$1_50epoch_with_model.pt

python3 train_cnn.py \
  --dataset_name $1 \
  --delim tab \
  --pred_len 12 \
  --encoder_h_dim_g 32 \
  --embedding_dim 32 \
  --num_layers 4 \
  --l2_loss_weight 1 \
  --dropout 0 \
  --batch_size 32 \
  --g_learning_rate 1e-3 \
  --print_every 200 \
  --num_epochs 100 \
  --checkpoint_name $1_100epoch \
  --checkpoint_every 50 \
  --output_dir save \
  --loader_num_workers 0 \
  --restore_from_checkpoint $2

echo "TESTING NOW epochs 100"
python3 evaluate_cnn.py --model_path save/$1_100epoch_with_model.pt

python3 train_cnn.py \
  --dataset_name $1 \
  --delim tab \
  --pred_len 12 \
  --encoder_h_dim_g 32 \
  --embedding_dim 32 \
  --num_layers 4 \
  --l2_loss_weight 1 \
  --dropout 0 \
  --batch_size 32 \
  --g_learning_rate 1e-4 \
  --print_every 500 \
  --num_epochs 200 \
  --checkpoint_name $1_200epoch \
  --checkpoint_every 50 \
  --output_dir save \
  --loader_num_workers 0 \
  --restore_from_checkpoint $2

echo "TESTING NOW epochs 200"
python3 evaluate_cnn.py --model_path save/$1_200epoch_with_model.pt
