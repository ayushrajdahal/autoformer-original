# Set the GPU device to be used
export CUDA_VISIBLE_DEVICES=1

# Run the first experiment with specific parameters
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the second experiment with different encoder layers
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the third experiment with different decoder layers
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the fourth experiment with more decoder layers
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the fifth experiment with different prediction length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the sixth experiment with shorter prediction length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the seventh experiment with more encoder layers
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the eighth experiment with different prediction length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the ninth experiment with different label length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the tenth experiment with shorter label length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the eleventh experiment with shorter sequence length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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

# Run the twelfth experiment with longer sequence length
python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/Energy/ \
  --data_path load_forecasting.csv \
  --model_id Energy_96_24 \
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