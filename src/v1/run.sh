#!/usr/bin/env bash

filename="lgb.py"

echo "Run:" $filename

date >> stdout.txt
date >> stderr.txt

python $filename >> stdout.txt 2>> stderr.txt

if [ $? -ne 0 ]
then
    echo "Run Failed!"
else
    echo "Run Success!"
fi