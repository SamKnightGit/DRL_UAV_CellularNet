import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for random_seed in [10, 222, 44, 505050, 1234]:
        model_dir_path = f"./experiment/adqn_seed_test/adqn_seed_{random_seed}"
        os.makedirs(model_dir_path, exist_ok=True)
        subprocess.call([
            "python",
            "run_adqn.py",
            "--num_base_stations=4",
            "--num_users=40",
            "--arena_width=100",
            "--num_workers=4",
            "--max_episodes=1000",
            "--learning_rate=0.0001",
            "--target_update_frequency=10",
            "--network_update_frequency=10",
            "--epsilon=0.80",
            "--discount_factor=0.9",
            "--test_model=False",
            f"--random_seed={random_seed}",
            f"--model_directory={model_dir_path}"
        ])