#!/bin/bash

for i in {60..110..10}
do
    python run_adqn.py --random_seed=$i
done
