import tensorflow as tf
from mobile_env import MobiEnvironment
import model
from os import path, chdir
import numpy as np
chdir("/home/samdknight/Documents/Edinburgh/Diss2019/DRL_UAV_CellularNet")


def check_weights(folder="./experiment/2020-01-02 00:10:04.814369", checkpoint=0):
    num_base_stations, num_users, arena_width = get_metrics_from_summary(folder)
    env = MobiEnvironment(num_base_stations, num_users, arena_width)
    state_space = env.observation_space_dim
    action_space = env.action_space_dim
    global_network = model.A3CNetwork(
        state_space=state_space,
        action_space=action_space
    )
    global_network.load_weights(path.join(folder, f"checkpoint_{checkpoint}.h5"))
    new_state, _, _, _ = env.step(4)
    action_log_prob, _ = global_network(
        tf.convert_to_tensor(np.ravel(new_state)[np.newaxis, :], dtype=tf.float32)
    )
    print(action_log_prob)
    action_prob = tf.nn.softmax(tf.squeeze(action_log_prob)).numpy()
    print(action_prob)

def get_metrics_from_summary(folder):
    num_base_stations, num_users, arena_width = 0, 0, 0
    with open(path.join(folder, "summary.txt")) as summ_file:
        for line in summ_file.readlines():
            split_line = line.strip().split()
            metric = " ".join(split_line[:-1])
            if metric == "Number of Base Stations:":
                num_base_stations = int(split_line[-1])
            elif metric == "Number of Users:":
                num_users = int(split_line[-1])
            elif metric == "Arena Width:":
                arena_width = int(split_line[-1])
    return num_base_stations, num_users, arena_width


if __name__ == "__main__":
    check_weights()