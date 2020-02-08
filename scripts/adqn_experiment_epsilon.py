import os
import subprocess
os.chdir(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    for epsilon in [0.10, 0.25, 0.50, 0.75, 1.0]:
        model_dir_path = f"./experiment/adqn_test_epsilon/adqn_epsilon_{epsilon}"
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
            f"--epsilon={epsilon}",
            "--discount_factor=0.9",
            "--test_model=False",
            "--random_seed=22",
            f"--model_directory={model_dir_path}"
        ])