import os
import sys
os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(".")

import mobile_env
import numpy as np
import copy

def test_bs_locations():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 100, random_seed=2)
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")

    _, _, _, info = mobi_env.step(0)
    print(f"Base Station: {info.bs_loc}")
    print(f"Users: {info.ue_loc}")

    mobi_env.reset()
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")

    mobi_env2 = mobile_env.MobiEnvironment(1, 1, 100, random_seed=2)
    print(f"Base Station: {mobi_env2.bsLoc}")
    print(f"Users: {mobi_env2.ueLoc}")

    _, _, _, info = mobi_env.step(3)
    print(f"Base Station: {info.bs_loc}")
    print(f"Users: {info.ue_loc}")

    mobi_env = mobile_env.MobiEnvironment(1, 1, 5, random_seed=22)
    print(f"Base Station: {mobi_env.bsLoc}")
    for _ in range(6):
        _, _, _, info = mobi_env.step(0)
        print(f"Base Station: {info.bs_loc}")   
    for _ in range(6):
        _, _, _, info = mobi_env.step(1)
        print(f"Base Station: {info.bs_loc}")
    for _ in range(6):
        _, _, _, info = mobi_env.step(2)
        print(f"Base Station: {info.bs_loc}")
    for _ in range(6):
        state, _, _, info = mobi_env.step(3)
        print(f"Base Station: {info.bs_loc}")
    print(state)
    

def test_state_space():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 10)
    print(mobi_env.state)
    new_state, reward, done, info = mobi_env.step(3)
    print(new_state.shape)

def test_new_reward():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 10, random_seed=22)
    _, reward, done, _ = mobi_env.step(4)
    print(f"Reward when not on top of user: {reward}")
    mobi_env.bsLoc = copy.deepcopy(mobi_env.ueLoc)
    mobi_env.bsLoc[0][2] = 10
    _, reward, done, _ = mobi_env.step(4)
    print(f"Reward when on top of user: {reward}")


def test_rewards():
    mobi_env = mobile_env.MobiEnvironment(1, 1, 5, random_seed=22)
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")
    done = False
    total_reward = 0
    while not done:
        _, reward, done, _ = mobi_env.step(4)
        total_reward += reward
    print(f"Total reward from random start: {total_reward}")
    mobi_env.reset()
    mobi_env.bsLoc = copy.deepcopy(mobi_env.ueLoc)
    mobi_env.bsLoc[0][2] = 10
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")
    done = False
    total_reward = 0
    while not done:
        _, reward, done, _ = mobi_env.step(4)
        total_reward += reward
    print(f"Total reward from start on user: {total_reward}")

    mobi_env.reset()
    mobi_env.bsLoc = copy.deepcopy(mobi_env.ueLoc)
    mobi_env.bsLoc[0][2] = 10
    mobi_env.bsLoc[0][1] = 4
    print(f"Base Station: {mobi_env.bsLoc}")
    print(f"Users: {mobi_env.ueLoc}")
    done = False
    total_reward = 0
    first_moves = 0
    while not done:
        if first_moves < 2:
            _, reward, done, _ = mobi_env.step(3)
            first_moves += 1
        else:
            _, reward, done, _ = mobi_env.step(4)
        total_reward += reward
    print(f"Total reward from move to user: {total_reward}")



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
    test_new_reward()