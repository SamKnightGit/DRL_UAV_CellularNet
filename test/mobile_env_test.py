import os
import sys
os.chdir("/home/sam/Documents/Dissertation/drones")
sys.path.append(".")

import mobile_env
import numpy as np

def test_bs_locations():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 100, random_seed=2)
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")
    # for i in range(5):
    #     get_action(i)
    #     mobi_env.step(i)
    #     print(mobi_env.bsLoc)
    for _ in range(10):
        _, reward, _, _ = mobi_env.step_test(3)
        print(f"Reward: {reward}")
        print(f"Base Station: {mobi_env.bsLoc}")
        print(f"Users: {mobi_env.ueLoc}")
    mobi_env.reset()
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")

    mobi_env2 = mobile_env.MobiEnvironment(1, 1, 100, random_seed=2)
    print(f"Base Station: {mobi_env2.bsLoc}")
    print(f"Users: {mobi_env2.ueLoc}")

def test_state_space():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 100)
    new_state, reward, done, info = mobi_env.step_test(3)
    print(new_state.shape)

def test_rewards():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 100, random_seed=2)
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")
    for _ in range(10):
        new_state, reward, _, _ = mobi_env.step(4)
        print(f"New State: {new_state}")
        print(f"Reward: {reward}")


def get_action(action):
    if action == 0.0:
        print("Right")
    elif action == 1.0:
        print("Left")
    elif action == 2.0:
        print("Up")
    elif action == 3.0:
        print("Down")
    elif action == 4.0:
        print("No Move")


if __name__ == "__main__":
    test_rewards()