export CUDA_VISIBLE_DEVICES=1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_3_weather \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1