#!/bin/bash

python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=10 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=20 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=30 
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=40 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=50 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=60 
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=70 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=80 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=90 &
sleep 5
python run_adqn.py --num_base_stations=4 --num_users=40 --num_workers=4 --max_episodes=1000 --epsilon=0.8 --random_seed=100 

