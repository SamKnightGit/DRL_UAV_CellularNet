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
MAX_GLOBAL_EP = 20
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TENSOR_SEED = 6

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
        self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
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
    def __init__(self, num_workers):
        self.network = A2CNet()
        self.update_barrier = Barrier(N_WORKERS + 1)        # Number of workers + coordinator which unblocks
        self.experience_queue = multiprocessing.Queue()
        self.workers = []
        for worker_index in range(num_workers):
            worker = Worker("W_{}".format(worker_index), self.network, self.update_barrier, self.experience_queue)
            self.workers.append(worker)

    def run(self):
        print(tf.get_collection(tf.GraphKeys.VARIABLES))
            

class Worker(object):
    def __init__(self, name, network, update_barrier, experience_queue):
        self.env = MobiEnvironment(N_BS, N_UE, AREA_W)#gym.make(GAME).unwrapped
        self.name = name
        self.network = network
        self.update_barrier = update_barrier
        self.experience_queue = experience_queue
        # self.buf_r_dissect_all_ep = []
        self.step_start = 0
        self.step_end = 0

    
    def work(self):
        pass


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

    start = time.time()
    
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        coordinator = Coordinator(num_workers=1)  # we only need its params
        coordinator.run()


    """
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker namei
            print "Creating worker ", i_name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    np.savez("train/Global_A_PARA_init", SESS.run(GLOBAL_AC.actor_params))

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, SESS.graph)
    
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
	
    end = time.time()
    print "Total time ", (end - start)
    """
