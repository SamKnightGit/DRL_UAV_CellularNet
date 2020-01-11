import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for num_workers in [1, 2, 4, 8, 16]:
        model_dir_path = f"./experiment/worker_test/workers_{num_workers}"
        os.makedirs(model_dir_path, exist_ok=True)
        subprocess.call([
            "python",
            "run_a3c.py",
            "--num_base_stations=1",
            "--num_users=1",
            "--max_episodes=800",
            f"--num_workers={num_workers}",
            "--random_seed=222",
            "--num_checkpoints=10",
            "--test_episodes=1",
            f"--model_directory={model_dir_path}"
        ])
