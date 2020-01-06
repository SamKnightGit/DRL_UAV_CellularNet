import agent
import model
import os
import tensorflow as tf
import click
import matplotlib.pyplot as plt
from datetime import datetime
from time import time, sleep
from tqdm import tqdm
from queue import Queue
from mobile_env import MobiEnvironment


@click.command()
@click.option('--num_base_stations', type=int, default=4)
@click.option('--num_users', type=int, default=40)
@click.option('--arena_width', type=int, default=100)
@click.option('--num_workers', type=int, default=1)
@click.option('--max_episodes', type=int, default=100)
@click.option('--learning_rate', type=float, default=1e-3)
@click.option('--network_update_frequency', type=int, default=50)
@click.option('--norm_clip_value', type=float, default=0.5)
@click.option('--num_checkpoints', type=int, default=10)
@click.option('--model_directory', type=click.Path(), default="")
@click.option('--test_model', type=bool, default=True)
@click.option('--test_episodes', type=int, default=100)
@click.option('--render_testing', type=bool, default=False)
@click.option('--random_seed', type=int, default=None)
def run_training(
        num_base_stations,
        num_users,
        arena_width,
        num_workers,
        max_episodes,
        learning_rate,
        network_update_frequency,
        norm_clip_value,
        num_checkpoints,
        model_directory,
        test_model,
        test_episodes,
        render_testing,
        random_seed):
    if random_seed is not None:
        tf.random.set_seed(random_seed)

    env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
    state_space = env.observation_space_dim
    action_space = env.action_space_dim
    global_network = model.A3CNetwork(
        state_space=state_space,
        action_space=action_space
    )

    if not model_directory:
        model_directory = os.path.join(
            "./experiment/",
            f"{datetime.now()}"
        )
    os.makedirs(model_directory, exist_ok=True)

    reward_queue = Queue()
    optimizer = tf.optimizers.RMSprop(learning_rate)
    workers = [
        agent.Worker(
            worker_index,
            global_network,
            num_base_stations,
            num_users,
            arena_width,
            max_episodes,
            optimizer,
            network_update_frequency,
            norm_clip_value,
            num_checkpoints,
            reward_queue,
            model_directory
        ) for worker_index in range(num_workers)
    ]
    start_time = time()
    for worker in workers:
        print(f"Starting Worker: {worker.name}")
        worker.start()
        sleep(0.1)

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

    write_summary(model_directory,
                  num_workers,
                  max_episodes,
                  learning_rate,
                  network_update_frequency,
                  time_taken,
                  num_base_stations,
                  num_users,
                  arena_width,
                  random_seed,
                  global_network,
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

            test_file_path = os.path.join(
                test_dir,
                f"checkpoint_{checkpoint}",
                f"test_checkpoint_{checkpoint}.txt"
            )
            os.makedirs(os.path.dirname(test_file_path))

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
    global_network = model.A3CNetwork(
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
        network_update_frequency,
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
        fp.write(f"Training Episodes:".ljust(35) + f"{max_episodes}\n")
        fp.write(f"Learning Rate:".ljust(35) + f"{learning_rate}\n")
        fp.write(f"Network Update Frequency:".ljust(35) + f"{network_update_frequency}\n")
        fp.write(f"Time Taken:".ljust(35) + f"{time_taken}\n")
        fp.write(f"Number of Base Stations:".ljust(35) + f"{num_base_stations}\n")
        fp.write(f"Number of Users:".ljust(35) + f"{num_users}\n")
        fp.write(f"Arena Width:".ljust(35) + f"{arena_width}\n")
        fp.write(f"Random Seed:".ljust(35) + f"{random_seed}\n")
        fp.write("Network Architecture:\n")
        global_network.summary(print_fn=lambda summ: fp.write(summ + "\n"))




if __name__ == "__main__":
    run_training()

