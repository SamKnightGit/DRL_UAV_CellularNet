import os
import shutil

for seed in range(10, 110, 10):
    path_to_seed_folder = f"/home/sam/Documents/Dissertation/drones/experiment/a3c_4_40_final/{seed}/"
    for checkpoint in range(10):
        checkpoint_path = os.path.join(path_to_seed_folder, f"checkpoint_{checkpoint}")
        shutil.copyfile(os.path.join(checkpoint_path, "model.h5"), os.path.join(path_to_seed_folder, f"checkpoint_{checkpoint}.h5"))
        shutil.rmtree(checkpoint_path)
