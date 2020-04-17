#!/bin/bash

for i in 10 20 30
do
	python run_adqn.py --max_episodes=10000 --random_seed=$i --annealing_episodes=2000 --model_directory="/home/sam/Documents/Dissertation/drones/experiment/adqn_meanSINR/random_$i"
done
