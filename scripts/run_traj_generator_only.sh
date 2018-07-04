
python3 train_generator_only.py \
  --dataset_name 'zara1' \
  --delim tab \
  --pred_len 8 \
  --encoder_h_dim_g 32 \
  --embedding_dim 16 \
  --num_layers 3 \
  --l2_loss_weight 1 \
  --dropout 0 \
  --batch_size 32 \
  --g_learning_rate 1e-3 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 20000 \
  --num_epochs 500 \
  --gpu_num 0 \
  --checkpoint_name gan_test_generator_only_remove_disc_z1 \
  --output_dir organized_cnn_models \
  --restore_from_checkpoint 1

