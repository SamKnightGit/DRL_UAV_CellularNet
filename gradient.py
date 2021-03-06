#from ue_mobility import *
from mobile_env import *
from copy import deepcopy
# from random import randint
import time
from itertools import product

FILE_NAME_APPEND = ""
OUTPUT_DIR = "gradient/"
OUTPUT_FILE_NAME = OUTPUT_DIR + "reward" + FILE_NAME_APPEND
N_BS = 4


def Choose_Act_Gradient(actual_env, s, n_step):
    virtual_env = deepcopy(actual_env)
    #BS remains but UE moves
    _, _, _, _ = virtual_env.step_test(624, False)
#    print np.equal(virtual_env.bsLoc, actual_env.bsLoc), virtual_env.bsLoc,"\n", actual_env.bsLoc,"\n"

    current_BS_sinr = virtual_env.channel.current_BS_sinr
    bs_loc = virtual_env.bsLoc
    ue_loc = virtual_env.ueLoc
    
    act_all_bs = np.zeros((len(bs_loc)))
    
    for i_bs in range(len(bs_loc)):
        dir_grad = np.zeros((4,1))
        dir_grad[0] = np.mean(current_BS_sinr[np.where(ue_loc[:,0] > bs_loc[i_bs][0])])
        dir_grad[1] = np.mean(current_BS_sinr[np.where(ue_loc[:,0] <= bs_loc[i_bs][0])])
        dir_grad[2] = np.mean(current_BS_sinr[np.where(ue_loc[:,1] > bs_loc[i_bs][1])])
        dir_grad[3] = np.mean(current_BS_sinr[np.where(ue_loc[:,1] <= bs_loc[i_bs][1])])
        act_all_bs[i_bs] = np.nanargmin(dir_grad)
    
    action = int(act_all_bs[3] + act_all_bs[2]*5 + act_all_bs[1]*(5**2) +  act_all_bs[0]*(5**3))
    
    #     print act_reward, "best action:", best_act
    return action

def Run_Test(reward_file_name):
    MAX_STEP = 10000
    #if reading mobility trace from file
    test_env = MobiEnvironment(N_BS, 40, 100, "read_trace", "./ue_trace_10k.npy")
    
    s = np.array([np.ravel(test_env.reset())])
    
    done = False
    step = 0
    
    outage_buf = []
    reward_buf = []
    sinr_all = []
    n_step_forward = 1
    reward_file_name = reward_file_name + str(n_step_forward)
    start_time = time.time()
    single_step_time = []
    while step <= MAX_STEP:
        before_step_time = time.time()
        action = Choose_Act_Gradient(test_env, s, n_step_forward)
        single_step_time.append(time.time() - before_step_time)
        
        s_, r, done, info = test_env.step_test(action, False)
        
        reward_buf.append(info[0])
	sinr_all.append(test_env.channel.current_BS_sinr)       
        if step % 500 == 0 or step == MAX_STEP:
            print "step ", step, " time ellipsed ", time.time() - start_time
            start_time = time.time()
            np.save(reward_file_name, reward_buf)
            np.save(OUTPUT_DIR + "time",single_step_time)
            np.save(OUTPUT_DIR + "sinr",sinr_all)
        # reset the environment every 2000 steps
        if step % 2000 == 0:
            s = np.array([np.ravel(test_env.reset())])
            #warm up in 500 steps
            for _ in range(500):
                action = Choose_Act_Gradient(test_env, s, n_step_forward)
                _, _, _, _ = test_env.step_test(action, False)
        else:
            s = np.array([np.ravel(s_)])
        
        step+=1

    np.save(reward_file_name, reward_buf)
    np.save(OUTPUT_DIR + "time",single_step_time)
    np.save(OUTPUT_DIR + "sinr",sinr_all)
if __name__ == "__main__":
    Run_Test(OUTPUT_FILE_NAME)


