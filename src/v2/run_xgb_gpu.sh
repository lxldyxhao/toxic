#!/usr/bin/env bash

python xgb_gpu.py -t toxic
python xgb_gpu.py -t severe_toxic
python xgb_gpu.py -t obscene
python xgb_gpu.py -t threat
python xgb_gpu.py -t insult
python xgb_gpu.py -t identity_hate