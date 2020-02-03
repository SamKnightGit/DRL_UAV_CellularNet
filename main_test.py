import time
import tensorflow as tf
import main
import os
import numpy as np
from mobile_env import MobiEnvironment

OUTPUT_FILE_NAME = "/test/A3C_median_SINR_"


def Load_AC_Net(tf_session, parameter_folder_name):
    """
    Load pre-trained A3C model for testing
    """
    file_name = "train/" + parameter_folder_name + "/Global_A_PARA.npz"
    files = np.load(file_name)

    a_params = files['arr_0']

    G_AC_TEST = main.ACNet('Global_Net')

    ops = []
    for idx, param in enumerate(a_params):
        ops.append(G_AC_TEST.a_params[idx].assign(param))
    tf_session.run(ops)
    return G_AC_TEST

# def Load_DPPO_Net():
#     """
#     Load pre-trained DDPO model for testing
#     """
#     SESS = tf.Session()
#
#     # file_name = "test/PI_PARA" + FILE_NAME_APPEND +".npz"
#     files = np.load(file_name)
#
#     pi_params = files['arr_0']
#
#     # G_PPO_TEST = PPONet()
#
#     ops = []
#     for idx, param in enumerate(pi_params): ops.append(G_PPO_TEST.pi_params[idx].assign(param))
#     SESS.run(ops)
#     return G_PPO_TEST

def Run_Test(g_test_net, reward_file_name, tf_session):
    #maximum training step
    MAX_STEP = 2000

    #Reading mobility trace from file
    test_env = MobiEnvironment(main.N_BS, 40, 100, "read_trace", "./ue_trace_10k.npy")
    # test_env = MobiEnvironment(main.N_BS, main.N_UE, main.AREA_W)
    #reset states
    s = np.array([np.ravel(test_env.reset())])

    done = False
    step = 0

    outage_buf = []
    reward_buf = []
    decomposed_reward_buf = []
    sinr_all = []
    time_all = []
    ue_loc_buf = []
    bs_loc_buf = []
    action_buf = []
    sinr_area_buf = []
    x = tf.argmax(g_test_net.a_prob, axis=1)
#    ue_walk_trace = []
    while step <= MAX_STEP:
        feed_dict = {g_test_net.s: s}
        start_time = time.time()
        action = tf_session.run(x, feed_dict=feed_dict)
        time_all.append(time.time()-start_time)
        s_, r, done, info = test_env.step_test(action, False)
        # s_, r, done, info = test_env.step(action, False)
        reward_buf.append(r)
        sinr_all.append(test_env.channel.current_BS_sinr)
        decomposed_reward_buf.append(info.r_dissect)
        outage_buf.append(info.outage_fraction)

        ue_loc_buf.append(info.ue_loc)
        bs_loc_buf.append(info.bs_loc)
        action_buf.append(info.bs_actions)
        if step % 500 == 0 or step == MAX_STEP:
            print "step ", step
            np.save(reward_file_name + "decomposed_reward", decomposed_reward_buf)
            np.save(reward_file_name + "reward", reward_buf)
            sinr_area_buf.append(test_env.channel.GetSinrInArea(info.bs_loc))
        np.save(reward_file_name + "sinr", sinr_all)
        np.save(reward_file_name + "time", time_all)

        # reset the environment every 2000 steps
        # if step % 2000 == 0:
        #     s = np.array([np.ravel(test_env.reset())])
        #     #warm up in 500 steps
        #     for _ in range(500):
        #         _, _, _, _ = test_env.step_test(action, False)
        #
        # else:
        s = np.array([np.ravel(s_)])
        
        step+=1

    np.save(reward_file_name + "reward", reward_buf)
    np.save(reward_file_name + "decomposed_reward", decomposed_reward_buf)
    np.save(reward_file_name + "sinr",sinr_all)
    np.save(reward_file_name + "time", time_all)
    np.save(reward_file_name + "outage_fraction", outage_buf)
    np.save(reward_file_name + "ue_location", ue_loc_buf)
    np.save(reward_file_name + "bs_location", bs_loc_buf)
    np.save(reward_file_name + "action", action_buf)
    np.save(reward_file_name + "sinr_area", sinr_area_buf)
#    np.save("ue_trace_10k", ue_walk_trace)


def test_network(test_architecture="AC", parameter_folder_name=""):
    tf_session = tf.Session()
    if test_architecture == "AC":
        test_net = Load_AC_Net(tf_session, parameter_folder_name)
    else: # default to AC
        test_net = Load_AC_Net(tf_session, parameter_folder_name)
    reward_file_name = "./test/" + parameter_folder_name + "/"
    try:
        os.makedirs(reward_file_name)
    except OSError:
        pass
    Run_Test(test_net, reward_file_name, tf_session)

if __name__ == "__main__":
    test_network(parameter_folder_name="A2C_median_SINR_200episodes_50step_rollout")


