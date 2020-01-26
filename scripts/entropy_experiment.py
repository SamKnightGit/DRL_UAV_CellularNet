import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for entropy in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 10, 100]:
        model_dir_path = f"./experiment/entropy_test/entropy_{entropy}"
        os.makedirs(model_dir_path, exist_ok=True)
        subprocess.call([
            "python",
            "run_a3c.py",
            "--num_base_stations=1",
            "--num_users=1",
            "--arena_width=5",
            "--max_episodes=5000",
            "--num_workers=10",
            "--random_seed=22",
            "--test_with_random_seed=True",
            "--num_checkpoints=10",
            "--test_episodes=1",
            f"--entropy_coefficient={entropy}",
            f"--model_directory={model_dir_path}"
        ])
