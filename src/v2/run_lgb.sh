#!/usr/bin/env bash

#  nohup sh run_lgb.sh >> lgb.out 2>&1 &

python lgb.py -t toxic
python lgb.py -t severe_toxic
python lgb.py -t obscene
python lgb.py -t threat
python lgb.py -t insult
python lgb.py -t identity_hate