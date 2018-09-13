#!/usr/bin/env bash

#  nohup sh run_xgb.sh >> xgb.out 2>&1 &

#python stacking_data.py -t toxic
#python stacking_data.py -t severe_toxic
#python stacking_data.py -t obscene
#python stacking_data.py -t threat
#python stacking_data.py -t insult
#python stacking_data.py -t identity_hate
#
#python bilstm_cnn_stacking.py -t 0
#python bilstm_cnn_stacking.py -t 1
#python bilstm_cnn_stacking.py -t 2
#python bilstm_cnn_stacking.py -t 3
#python bilstm_cnn_stacking.py -t 4

python gru_fasttext_stacking.py -t 0
python gru_fasttext_stacking.py -t 1
python gru_fasttext_stacking.py -t 2
python gru_fasttext_stacking.py -t 3
python gru_fasttext_stacking.py -t 4