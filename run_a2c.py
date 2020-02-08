import a2c_agent as agent
import model
import os
import tensorflow as tf
import numpy as np
import click
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import time, sleep
from tqdm import tqdm
from queue import Queue
from mobile_env_original import MobiEnvironment, MAXSTEP


@click.command()
@click.option('--num_base_stations', type=int, default=1)
@click.option('--num_users', type=int, default=1)
@click.option('--arena_width', type=int, default=100)
@click.option('--num_workers', type=int, default=1)
@click.option('--max_episodes', type=int, default=100)
@click.option('--timesteps_per_rollout', type=int, default=50)
@click.option('--learning_rate', type=float, default=10e-4)
@click.option('--network_update_frequency', type=int, default=50)
@click.option('--entropy_coefficient', type=float, default=0.01)
@click.option('--norm_clip_value', type=float, default=None)
@click.option('--num_checkpoints', type=int, default=10)
@click.option('--model_directory', type=click.Path(), default="")
@click.option('--test_model', type=bool, default=True)
@click.option('--test_episodes', type=int, default=100)
@click.option('--render_testing', type=bool, default=False)
@click.option('--random_seed', type=int, default=None)
@click.option('--test_with_random_seed', type=bool, default=False)
def run_training(
        num_base_stations,
        num_users,
        arena_width,
        num_workers,
        max_episodes,
        timesteps_per_rollout,
        learning_rate,
        network_update_frequency,
        entropy_coefficient,
        norm_clip_value,
        num_checkpoints,
        model_directory,
        test_model,
        test_episodes,
        render_testing,
        random_seed,
        test_with_random_seed):
    if random_seed is not None:
        tf.random.set_seed(random_seed)
        np.random_seed(random_seed)

    # env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
    env = MobiEnvironment(num_base_stations, num_users, arena_width)
    state_space = env.observation_space_dim
    action_space = env.action_space_dim
    global_network = model.A3CNetwork(
        state_space=state_space,
        action_space=action_space,
        entropy_coefficient=entropy_coefficient
    )

    if not model_directory:
        model_directory = os.path.join(
            "./experiment/",
            f"a2c_{datetime.now()}"
        )
    os.makedirs(model_directory, exist_ok=True)

    optimizer = tf.optimizers.Adam(learning_rate)
    coordinator = agent.Coordinator(
        global_network,
        num_workers,
        num_base_stations,
        num_users,
        arena_width,
        timesteps_per_rollout,
        MAXSTEP,
        max_episodes,
        num_checkpoints,
        norm_clip_value,
        optimizer,
        random_seed,
        model_directory
    )
    
    start_time = time()
    print(f"Starting A2C Coordinator!")
    coordinator.run()

    end_time = time()
    time_taken = end_time - start_time

    write_summary(model_directory,
                  num_workers,
                  max_episodes,
                  learning_rate,
                  network_update_frequency,
                  entropy_coefficient,
                  norm_clip_value,
                  time_taken,
                  num_base_stations,
                  num_users,
                  arena_width,
                  random_seed,
                  global_network,
                  filename="summary.txt")

    if test_model:
        test_dir = os.path.join(model_directory, "test")
        os.makedirs(test_dir)
        print("Running tests with checkpoint policies...")
        for checkpoint in tqdm(range(num_checkpoints)):
            model_file_path = os.path.join(
                model_directory,
                f"checkpoint_{checkpoint}.h5",
            )

            test_file_path = os.path.join(
                test_dir,
                f"test_checkpoint_{checkpoint}.txt"
            )

            if not test_with_random_seed:
                random_seed = None
                
            run_testing(
                num_base_stations,
                num_users,
                arena_width,
                test_episodes,
                model_file_path,
                test_file_path,
                render_testing,
                random_seed
            )


def run_testing(
        num_base_stations,
        num_users,
        arena_width,
        max_episodes,
        model_file,
        test_file_name,
        render,
        random_seed):
    if random_seed is not None:
        tf.random.set_seed(random_seed)

    # env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
    env = MobiEnvironment(num_base_stations, num_users, arena_width, "read_trace", "./ue_trace_10k.npy")
    state_space = env.observation_space_dim
    action_space = env.action_space_dim
    global_network = model.A3CNetwork(
        state_space=state_space,
        action_space=action_space
    )

    global_network.load_weights(
        model_file
    )

    worker = agent.TestWorker(
        num_base_stations,
        num_users,
        arena_width,
        global_network,
        max_episodes,
        test_file_name,
        render=render,
        random_seed=random_seed
    )
    worker.run()


def write_summary(
        model_directory,
        num_workers,
        max_episodes,
        learning_rate,
        network_update_frequency,
        entropy_coefficient,
        norm_clip_value,
        time_taken,
        num_base_stations,
        num_users,
        arena_width,
        random_seed,
        global_network,
        filename="summary.txt"):
    filepath = os.path.join(model_directory, filename)
    with open(filepath, "w+") as fp:
        fp.write("Number of Workers:".ljust(35) + f"{num_workers}\n")
        fp.write("Training Episodes:".ljust(35) + f"{max_episodes}\n")
        fp.write("Learning Rate:".ljust(35) + f"{learning_rate}\n")
        fp.write("Network Update Frequency:".ljust(35) + f"{network_update_frequency}\n")
        fp.write("Entropy Coefficient:".ljust(35) + f"{entropy_coefficient}\n")
        fp.write("Norm Clip Value:".ljust(35) + f"{norm_clip_value}\n")
        fp.write("Time Taken:".ljust(35) + f"{time_taken}\n")
        fp.write("Formatted Time:".ljust(35) + f"{timedelta(seconds=time_taken)}\n")
        fp.write("Number of Base Stations:".ljust(35) + f"{num_base_stations}\n")
        fp.write("Number of Users:".ljust(35) + f"{num_users}\n")
        fp.write("Arena Width:".ljust(35) + f"{arena_width}\n")
        fp.write("Random Seed:".ljust(35) + f"{random_seed}\n")
        fp.write("Network Architecture:\n")
        global_network.summary(print_fn=lambda summ: fp.write(summ + "\n"))




if __name__ == "__main__":
    run_training()

