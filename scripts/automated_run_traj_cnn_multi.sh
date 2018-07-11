# use it from the scripts folder
# usage run_traj_cnn.sh dataset_name restore
export PYTHONPATH=`cd .. && pwd`

for i in {1..3}
do
python3 train_generator_only.py \
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
  --checkpoint_every 100 \
  --print_every 200 \
  --num_epochs 50 \
  --checkpoint_name organized_cnn_$1_4lay_12 \
  --output_dir organized_cnn_models_multi \
  --loader_num_workers 8 \
  --restore_from_checkpoint $2

echo "TESTING NOW epochs 50" 
python3 evaluate_generator_only.py --model_path organized_cnn_models_multi/organized_cnn_$1_4lay_12_with_model.pt

python3 train_generator_only.py \
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
  --checkpoint_every 100 \
  --print_every 200 \
  --num_epochs 100 \
  --checkpoint_name organized_cnn_$1_4lay_12 \
  --output_dir organized_cnn_models_multi \
  --loader_num_workers 8 \
  --restore_from_checkpoint 1

echo "TESTING NOW epochs 100"
python3 evaluate_generator_only.py --model_path organized_cnn_models_multi/organized_cnn_$1_4lay_12_with_model.pt


python3 train_generator_only.py \
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
  --checkpoint_every 100 \
  --print_every 200 \
  --num_epochs 150 \
  --checkpoint_name organized_cnn_$1_4lay_12 \
  --output_dir organized_cnn_models_multi \
  --loader_num_workers 8 \
  --restore_from_checkpoint 1

echo "TESTING NOW epochs 150"
python3 evaluate_generator_only.py --model_path organized_cnn_models_multi/organized_cnn_$1_4lay_12_with_model.pt

python3 train_generator_only.py \
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
  --checkpoint_every 100 \
  --print_every 200 \
  --num_epochs 200 \
  --checkpoint_name organized_cnn_$1_4lay_12 \
  --output_dir organized_cnn_models_multi \
  --loader_num_workers 8 \
  --restore_from_checkpoint 1

echo "TESTING NOW epochs 200"
python3 evaluate_generator_only.py --model_path organized_cnn_models_multi/organized_cnn_$1_4lay_12_with_model.pt

done








