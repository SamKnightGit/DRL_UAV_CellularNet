#!/bin/bash

for i in 10 20 30
do
	python run_adqn.py --max_episodes=10000 --random_seed=$i --clipped_reward=True --model_directory="/home/sam/Documents/Dissertation/drones/experiment/adqn_meanSINR/1bs_10ue/clipped_reward_$i"
done
