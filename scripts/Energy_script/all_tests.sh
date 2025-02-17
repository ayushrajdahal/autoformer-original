export CUDA_VISIBLE_DEVICES=1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_1_default \
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

# python3 -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/Energy/ \
#   --data_path load_forecasting.csv \
#   --model_id 2_2_sincos \
#   --model Autoformer \
#   --data Energy \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 16 \
#   --dec_in 16 \
#   --c_out 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --embed sincos

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_4_48seqlen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 48 \
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

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_4_144seqlen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 144 \
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

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_5_24labellen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_5_72labellen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 72 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_6_12predlen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_6_48predlen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_6_72predlen \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_7_1encoder \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_7_3encoders \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_8_2decoders \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 2_8_3decoders \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 3 \
  --factor 3 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 3_2train_1val_2test \
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
  --itr 1 \
  --train_yrs 2 \
  --val_yrs 1 \
  --test_yrs 2

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 4_1train_1val_3test \
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
  --itr 1 \
  --train_yrs 1 \
  --val_yrs 1 \
  --test_yrs 3

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 6_1factor \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 6_2factor \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 2 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1

python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id 6_5factor \
  --model Autoformer \
  --data Energy \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 5 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 16 \
  --des 'Exp' \
  --itr 1