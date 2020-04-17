#!/bin/bash

for i in {10..110..10}
do
    python run_a3c.py --random_seed=$i
done
