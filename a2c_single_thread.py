import multiprocessing
import threading
import tensorflow as tf
import numpy as np
#import gym
import os
import shutil
import matplotlib.pyplot as plt
from mobile_env import *
from util import Barrier
import time

#matplotlib.use('Agg')

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 4#multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
TIMESTEPS_PER_ROLLOUT = 50
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TENSOR_SEED = 6
EXPERIMENT_NAME = "A2C_median_SINR_" + str(MAX_GLOBAL_EP) + "episodes_" + str(TIMESTEPS_PER_ROLLOUT) + "step_rollout"
try:
    os.makedirs("./train/" + EXPERIMENT_NAME)
except OSError:
    pass

N_BS = 4
N_UE = 40
AREA_W = 100 #width of the playground
env = MobiEnvironment(N_BS, N_UE, AREA_W)#gym.make(GAME)

N_S = env.observation_space_dim#number of state
N_A = env.action_space_dim

class A2CNet(object):
    def __init__(self):
        self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
        self.a_prob, self.v, self.actor_params, self.critic_params = self._build_net_mlp()
        self.a_his = tf.placeholder(tf.int32, [None,], 'A')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
        
        td = tf.subtract(self.v_target, self.v, name='TD_error')
        with tf.name_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(td))
    
        with tf.name_scope('actor_loss'):
            log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * 
                        tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)

            exp_v = log_prob * tf.stop_gradient(td)
            entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keep_dims=True)  # encourage exploration
            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.actor_loss = tf.reduce_mean(-self.exp_v)
        
        with tf.name_scope('gradients'):
            self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
            self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)
    
        with tf.name_scope('update_networks'):
            self.update_actor_op = OPT_A.apply_gradients(list(zip(self.actor_grads, self.actor_params)))
            self.update_critic_op = OPT_C.apply_gradients(list(zip(self.critic_grads, self.critic_params)))


    def _build_net_mlp(self):
        print "build MLP net"
        w_init = tf.random_normal_initializer(0., .1, seed = TENSOR_SEED)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 200, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        return a_prob, v, actor_params, critic_params

    
    def update_network(self, feed_dict):
        SESS.run([self.update_actor_op, self.update_critic_op], feed_dict)

    def choose_action(self, s):  
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Coordinator(object):
    def __init__(self, num_workers, timesteps_per_rollout, timesteps_per_episode, num_episodes):
        self.network = A2CNet()
        self.workers = []
        for worker_index in range(num_workers):
            worker = Worker("W_{}".format(worker_index), self.network, timesteps_per_rollout)
            self.workers.append(worker)
        self.timesteps_per_rollout = timesteps_per_rollout
        self.timesteps_per_episode = timesteps_per_episode
        self.num_episodes = num_episodes

    def run(self):
        rollouts_per_episode = self.timesteps_per_episode / self.timesteps_per_rollout

        for episode in range(self.num_episodes):
            for rollout in range(rollouts_per_episode):
                buffer_state, buffer_action, buffer_value_target = [], [], []
                for worker in self.workers:
                    print("Processing in worker {}, rollout {} of episode {}".format(worker.name, rollout, episode))
                    states, actions, targets = worker.work()
                    buffer_state += states
                    buffer_action += actions
                    buffer_value_target += targets

                buffer_state = np.vstack(buffer_state)
                buffer_action = np.array(buffer_action)
                buffer_value_target = np.vstack(buffer_value_target)

                print("Buffer State: {}".format(buffer_state))
                print("Buffer Action: {}".format(buffer_action))
                print("Buffer VT: {}".format(buffer_value_target))

                feed_dict = {
                    self.network.s: buffer_state,
                    self.network.a_his: buffer_action,
                    self.network.v_target: buffer_value_target
                }
                self.network.update_network(feed_dict)

        np.save("train/" + EXPERIMENT_NAME + "/Global_return", GLOBAL_RUNNING_R)
        np.savez("train/" + EXPERIMENT_NAME + "/Global_A_PARA", SESS.run(self.network.actor_params))
        

class Worker(object):
    global GLOBAL_RUNNING_R

    def __init__(self, name, network, timesteps_per_rollout):
        self.env = MobiEnvironment(N_BS, N_UE, AREA_W)
        self.name = name
        self.network = network
        self.timesteps_per_rollout = timesteps_per_rollout
        self.ep_r = 0

    def reset_worker(self):
        self.ep_r = 0
        self.env.reset()

    def work(self):
        buffer_state, buffer_action, buffer_reward = [], [], []
        state = np.ravel(np.array(self.env.state))
        for _ in range(self.timesteps_per_rollout):
            action = self.network.choose_action(state)
            state_next, reward, done, _ = self.env.step(action)
            self.ep_r += reward
            state_next = np.ravel(state_next)
            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append(reward)
            state = state_next

        if done:
            value_estimate = 0
            if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                GLOBAL_RUNNING_R.append(self.ep_r)
            else:
                GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * self.ep_r)
            self.reset_worker()
        else:
            value_estimate = SESS.run(
                self.network.v, 
                {self.network.s: state_next[np.newaxis, :]}
            )[0, 0]
        buffer_value_target = []

        for r in buffer_reward[::-1]:
            value_estimate = r + GAMMA * value_estimate
            buffer_value_target.append(value_estimate)

        buffer_value_target.reverse()

        return buffer_state, buffer_action, buffer_value_target


if __name__ == "__main__":
    print ">>>>>>>>>>>>>>>>A3C SIM INFO>>>>>>>>>>>>>>>>>>>>"
    print "tensor seed: ", TENSOR_SEED
    print "N_S", N_S
    print "N_A", N_A
    print "LR_C", LR_C
    print "N_BS", N_BS
    print "N_UE", N_UE
    print "AREA_W", AREA_W
    print "Num of episodes", MAX_GLOBAL_EP
    print ">>>>>>>>>>>>>>>>>>>>SIM INFO(end)>>>>>>>>>>>>>>>"
    
    SESS = tf.Session()
    with open("train/" + EXPERIMENT_NAME + "/experiment_summary.txt", "w+") as f:
        f.write("tensor seed: " + str(TENSOR_SEED) + "\n")
        f.write("N_S " + str(N_S) + "\n")
        f.write("N_A " + str(N_A) + "\n")
        f.write("LR_C " + str(LR_C) + "\n")
        f.write("N_BS " + str(N_BS) + "\n")
        f.write("N_UE " + str(N_UE) + "\n")
        f.write("AREA_W " + str(AREA_W) + "\n")
        f.write("Num of episodes " + str(MAX_GLOBAL_EP) + "\n")
        f.write(">>>>>>>>>>>>>>>>>>>>SIM INFO(end)>>>>>>>>>>>>>>>")

    start = time.time()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    coordinator = Coordinator(
        num_workers=4,
        timesteps_per_rollout=TIMESTEPS_PER_ROLLOUT,
        timesteps_per_episode=MAXSTEP,
        num_episodes=MAX_GLOBAL_EP
    )  # we only need its params
       
    SESS.run(tf.global_variables_initializer())
    np.savez("train/" + EXPERIMENT_NAME + "/Global_A_PARA_init", SESS.run(coordinator.network.actor_params))

    coordinator.run()
    end = time.time()
    with open("train/" + EXPERIMENT_NAME + "/time_taken.txt", "w+") as f:
        f.write("Total time taken" + str(end-start))
