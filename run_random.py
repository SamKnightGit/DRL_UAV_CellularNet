from mobile_env_original import MobiEnvironment
import click
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

@click.command()
@click.option('--num_base_stations', type=int, default=4)
@click.option('--num_users', type=int, default=40)
@click.option('--arena_width', type=int, default=100)
@click.option('--max_episodes', type=int, default=100)
def run_random(
    num_base_stations,
    num_users,
    arena_width,
    max_episodes):
    env = MobiEnvironment(num_base_stations, num_users, arena_width)
    action_space = env.action_space_dim
    episode = 0
    meanSINR = []
    nOutages = []
    rewards = []
    now = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    model_directory = os.path.join(
            "./experiment/",
            "random_agent",
            f"{now}"
        )
    os.makedirs(model_directory)
    for episode in tqdm(range(max_episodes)):
        env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = np.random.choice(action_space)
            _, reward, done, info = env.step(action)
            if done:
                reward = -1
            ep_reward += reward
            meanSINR.append(info[2])
            nOutages.append(info[3])
        rewards.append(ep_reward)

    np.save(os.path.join(model_directory, "meanSINR.npy"), meanSINR)
    np.save(os.path.join(model_directory, "nOutages.npy"), nOutages)
    np.save(os.path.join(model_directory, "rewards.npy"), rewards)

    
if __name__ == "__main__":
    run_random()