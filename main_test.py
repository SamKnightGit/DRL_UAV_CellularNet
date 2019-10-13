import time
TEST_ALGO = "A3C"

FILE_NAME_APPEND = ""
OUTPUT_FILE_NAME = "test/" + FILE_NAME_APPEND + '_'

def Load_AC_Net():
    """
    Load pre-trained A3C model for testing
    """
    file_name = "train/Global_A_PARA" + FILE_NAME_APPEND +".npz"
    files = np.load(file_name)

    a_params = files['arr_0']

    G_AC_TEST = ACNet('Global_Net')

    ops = []
    for idx, param in enumerate(a_params): ops.append(G_AC_TEST.a_params[idx].assign(param))
    SESS.run(ops)
    return G_AC_TEST

def Load_DPPO_Net():
    """
    Load pre-trained DDPO model for testing
    """

    file_name = "test/PI_PARA" + FILE_NAME_APPEND +".npz"
    files = np.load(file_name)

    pi_params = files['arr_0']

    G_PPO_TEST = PPONet()

    ops = []
    for idx, param in enumerate(pi_params): ops.append(G_PPO_TEST.pi_params[idx].assign(param))
    SESS.run(ops)
    return G_PPO_TEST

def Run_Test(g_test_net, reward_file_name):
    #maximum training step
    MAX_STEP = 10000

    #Reading mobility trace from file
    test_env = MobiEnvironment(N_BS, 40, 100, "read_trace", "./ue_trace_10k.npy")

    #reset states
    s = np.array([np.ravel(test_env.reset())])

    done = False
    step = 0

    outage_buf = []
    reward_buf = []
    decomposed_reward_buf = []
    sinr_all = []
    time_all = []
    x = tf.argmax(g_test_net.a_prob, axis = 1)
#    ue_walk_trace = []
    while step <= MAX_STEP:       
        feed_dict = { g_test_net.s:s }
        start_time = time.time()
        action = SESS.run(x, feed_dict=feed_dict)
        time_all.append(time.time()-start_time)

        s_, r, done, info = test_env.step_test(action, False)
        # s_, r, done, info = test_env.step(action, False)
        reward_buf.append(r)
     	sinr_all.append(test_env.channel.current_BS_sinr)   
        decomposed_reward_buf.append(info[0])
        
#        ue_walk_trace.append(info[2])
        if step % 500 == 0 or step == MAX_STEP:
            print "step ", step
            np.save(reward_file_name + "decomposed_reward", decomposed_reward_buf)
            np.save(reward_file_name + "reward", reward_buf)

        np.save(reward_file_name +"sinr", sinr_all)
        np.save(reward_file_name + "time", time_all)
        #            np.save("ue_trace_10k", ue_walk_trace)

        # reset the environment every 2000 steps
        if step % 2000 == 0:
            s = np.array([np.ravel(test_env.reset())])
            #warm up in 500 steps
            for _ in range(500):
                _, _, _, _ = test_env.step_test(action, False)
        else:
            s = np.array([np.ravel(s_)])
        
        step+=1

    np.save(reward_file_name + "reward", reward_buf)
    np.save(reward_file_name + "decomposed_reward", decomposed_reward_buf)
    np.save(reward_file_name + "sinr",sinr_all)
    np.save(reward_file_name + "time", time_all)
#    np.save("ue_trace_10k", ue_walk_trace)

if __name__ == "__main__":
    if TEST_ALGO == "A3C":
        from main import *
        SESS = tf.Session()
        
        test_net = Load_AC_Net()
    elif TEST_ALGO == "DPPO":
        from dppo_main import *
        SESS = tf.Session()
        
        test_net = Load_DPPO_Net()

    Run_Test(test_net, OUTPUT_FILE_NAME)


