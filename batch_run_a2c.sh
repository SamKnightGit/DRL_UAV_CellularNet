#!/bin/bash

for i in {10..110..10}
do
    python run_a2c.py --random_seed=$i
done
