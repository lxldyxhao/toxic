#!/usr/bin/env bash

#  nohup sh run_xgb.sh >> xgb.out 2>&1 &

python xgb.py -t toxic
python xgb.py -t severe_toxic
python xgb.py -t obscene
python xgb.py -t threat
python xgb.py -t insult
python xgb.py -t identity_hate