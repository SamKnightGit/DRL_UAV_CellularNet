import adqn_agent as agent
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
from mobile_env import MobiEnvironment

@click.command()
@click.option('--num_base_stations', type=int, default=1)
@click.option('--num_users', type=int, default=1)
@click.option('--arena_width', type=int, default=100)
@click.option('--num_workers', type=int, default=1)
@click.option('--max_episodes', type=int, default=100)
@click.option('--learning_rate', type=float, default=10e-4)
@click.option('--target_update_frequency', type=int, default=5)
@click.option('--network_update_frequency', type=int, default=50)
@click.option('--epsilon', type=float, default=0.10)
@click.option('--epsilon_annealing_strategy', type=str, default="linear")
@click.option('--discount_factor', type=float, default=0.95)
@click.option('--norm_clip_value', type=float, default=None)
@click.option('--num_checkpoints', type=int, default=10)
@click.option('--model_directory', type=click.Path(), default="")
@click.option('--test_model', type=bool, default=True)
@click.option('--test_episodes', type=int, default=100)
@click.option('--render_testing', type=bool, default=False)
@click.option('--random_seed', type=int, default=None)
@click.option('--test_with_random_seed', type=bool, default=False)
@click.option('--save', type=bool, default=True)
def run_training(
        num_base_stations,
        num_users,
        arena_width,
        num_workers,
        max_episodes,
        learning_rate,
        target_update_frequency,
        network_update_frequency,
        epsilon,
        epsilon_annealing_strategy,
        discount_factor,
        norm_clip_value,
        num_checkpoints,
        model_directory,
        test_model,
        test_episodes,
        render_testing,
        random_seed,
        test_with_random_seed,
        save):
    if random_seed is not None:
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
    state_space = env.observation_space_dim
    action_space = env.action_space_dim

    main_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )

    target_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )
    target_network.set_weights(main_network.get_weights())

    if not model_directory:
        model_directory = os.path.join(
            "./experiment/",
            f"adqn_{datetime.now()}"
        )
    if save:
        os.makedirs(model_directory, exist_ok=True)

    reward_queue = Queue()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    workers = [
        agent.Worker(
            worker_index,
            num_base_stations,
            num_users,
            arena_width,
            random_seed,
            max_episodes,
            optimizer,
            target_network,
            main_network,
            target_update_frequency,
            network_update_frequency,
            epsilon,
            epsilon_annealing_strategy,
            discount_factor,
            norm_clip_value,
            num_checkpoints,
            reward_queue,
            model_directory,
            save
        ) for worker_index in range(num_workers)
    ]
    start_time = time()
    for worker in workers:
        print(f"Starting Worker: {worker.name}")
        worker.start()

    moving_average_rewards = []
    while True:
        reward = reward_queue.get()
        if reward is not None:
            moving_average_rewards.append(reward)
        else:
            break

    for worker in workers:
        worker.join()

    end_time = time()
    time_taken = end_time - start_time

    if save:
        write_summary(model_directory,
                      num_workers,
                      max_episodes,
                      learning_rate,
                      target_update_frequency,
                      network_update_frequency,
                      epsilon,
                      discount_factor,
                      norm_clip_value,
                      time_taken,
                      num_base_stations,
                      num_users,
                      arena_width,
                      random_seed,
                      main_network,
                      filename="summary.txt")
        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average reward')
        plt.xlabel('Episode')
        plt.savefig(os.path.join(model_directory, 'Moving_Average.png'))

        if test_model:
            test_dir = os.path.join(model_directory, "test")
            os.makedirs(test_dir)
            print("Running tests with checkpoint policies...")
            for checkpoint in tqdm(range(num_checkpoints + 1)):
                if checkpoint == num_checkpoints:
                    model_file_path = os.path.join(
                        model_directory,
                        "best_model.h5"
                    )
                    checkpoint = "best"
                else:
                    model_file_path = os.path.join(
                        model_directory,
                        f"checkpoint_{checkpoint}",
                        "model.h5"
                    )

                if not os.path.exists(model_file_path):
                    break

                test_file_path = os.path.join(
                    test_dir,
                    f"checkpoint_{checkpoint}",
                    f"test_checkpoint_{checkpoint}.txt"
                )
                os.makedirs(os.path.dirname(test_file_path))

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

    env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
    state_space = env.observation_space_dim
    action_space = env.action_space_dim
    global_network = model.ADQNetwork(
        state_space=state_space,
        action_space=action_space
    )

    global_network.load_weights(
        model_file
    )

    worker = agent.TestWorker(
        global_network,
        num_base_stations,
        num_users,
        arena_width,
        max_episodes,
        test_file_name,
        render=render
    )
    worker.start()
    worker.join()


def write_summary(
        model_directory,
        num_workers,
        max_episodes,
        learning_rate,
        target_update_frequency,
        network_update_frequency,
        epsilon,
        discount_factor,
        norm_clip_value,
        time_taken,
        num_base_stations,
        num_users,
        arena_width,
        random_seed,
        main_network,
        filename="summary.txt"):
    filepath = os.path.join(model_directory, filename)
    with open(filepath, "w+") as fp:
        fp.write("Number of Workers:".ljust(35) + f"{num_workers}\n")
        fp.write("Training Episodes:".ljust(35) + f"{max_episodes}\n")
        fp.write("Learning Rate:".ljust(35) + f"{learning_rate}\n")
        fp.write("Target Update Frequency (eps):".ljust(35) + f"{target_update_frequency}\n")
        fp.write("Network Update Frequency:".ljust(35) + f"{network_update_frequency}\n")
        fp.write("Epsilon:".ljust(35) + f"{epsilon}\n")
        fp.write("Discount Factor:".ljust(35) + f"{discount_factor}\n")
        fp.write("Norm Clip Value:".ljust(35) + f"{norm_clip_value}\n")
        fp.write("Time Taken:".ljust(35) + f"{time_taken}\n")
        fp.write("Formatted Time:".ljust(35) + f"{timedelta(seconds=time_taken)}\n")
        fp.write("Number of Base Stations:".ljust(35) + f"{num_base_stations}\n")
        fp.write("Number of Users:".ljust(35) + f"{num_users}\n")
        fp.write("Arena Width:".ljust(35) + f"{arena_width}\n")
        fp.write("Random Seed:".ljust(35) + f"{random_seed}\n")
        fp.write("Network Architecture:\n")
        main_network.summary(print_fn=lambda summ: fp.write(summ + "\n"))


if __name__ == "__main__":
    run_training()


