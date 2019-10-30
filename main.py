import multiprocessing
import threading
import tensorflow as tf
import numpy as np
#import gym
import os
import shutil
import matplotlib.pyplot as plt
from mobile_env import *
import time

#matplotlib.use('Agg')

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 4#multiprocessing.cpu_count()
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
TENSOR_SEED = 6
CNN_NUM_FILTERS = 10
CNN_KERNEL_SIZE = 5

N_BS = 4
N_UE = 40
AREA_W = 100 #width of the playground
env = MobiEnvironment(N_BS, N_UE, AREA_W) #gym.make(GAME)
#env.plot_sinr_map()

N_S = env.observation_space_dim #number of state
N_A = env.action_space_dim

class ACNet(object):
    def __init__(self, is_global=False):
        
        if is_global:   # get global network
            self.s = tf.Variable(tf.float32, [None, N_S], 'S')
            self.actor_model, self.actor_optimizer, self.critic_model, self.critic_optimizer = self._build_net_mlp()
        else:   # local net, calculate losses
            self.s = tf.Variable(tf.float32, [None, N_S], 'S')
            self.a_his = tf.Variable(tf.int32, [None, ], 'A')
            self.v_target = tf.Variable(tf.float32, [None, 1], 'Vtarget')
            
            self.actor_model, self.actor_optimizer, self.critic_model, self.critic_optimizer = self._build_net_mlp()
            
            td = tf.subtract(self.v_target, self.critic_model, name='TD_error')
            with tf.name_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(td))
        
            with tf.name_scope('actor_loss'):
                log_prob = tf.reduce_sum(tf.math.log(self.actor_model + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                exp_v = log_prob * tf.stop_gradient(td)
                entropy = -tf.reduce_sum(self.actor_model * tf.math.log(self.actor_model + 1e-5),
                                            axis=1, keep_dims=True)  # encourage exploration
                self.exp_v = ENTROPY_BETA * entropy + exp_v
                self.actor_loss = tf.reduce_mean(-self.exp_v)
            
            with tf.name_scope('local_grad'):
                self.actor_gradients = tf.gradients(self.actor_loss, self.actor_model.trainable_variables)
                self.critic_gradients = tf.gradients(self.critic_loss, self.critic_model.trainable_variables)



    def _build_net_mlp(self):
        print("build MLP net")
        w_init = tf.random_normal_initializer(0., .1, seed = TENSOR_SEED)
        actor_model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, tf.nn.relu6, kernel_initializer=w_init, input_shape=self.s),
            tf.keras.layers.Dense(200, tf.nn.relu6, kernel_initializer=w_init),
            tf.keras.layers.Dense(N_A, tf.nn.softmax, kernel_initializer=w_init)
        ])
        actor_optimizer = tf.keras.optimizers.RMSprop(learning_rate=LR_A)
        actor_model.compile(actor_optimizer)

        critic_model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, tf.nn.relu6, kernel_initializer=w_init, input_shape=self.s),
            tf.keras.layers.Dense(200, tf.nn.relu6, kernel_initializer=w_init),
            tf.keras.layers.Dense(1, kernel_initializer=w_init)
        ])
        critic_optimizer = tf.keras.optimzers.RMSprop(learning_rate=LR_C)
        critic_model.compile(critic_optimizer)

        return actor_model, actor_optimizer, critic_model, critic_optimizer


    def update_global(self, global_network):
        global_network.actor_optimizer.apply_gradients(zip(self.actor_gradients, global_network.actor_model.trainable_variables))
        global_network.critic_optimizer.apply_gradients(zip(self.critic_gradients, global_network.critic_model.trainable_variables))
    
    @tf.function
    def pull_from_global(self, global_network):  # run by a local
        actor_params = [local_params.assign(global_params) for local_params, global_params in 
                            zip(self.actor_model.trainable_variables, global_network.actor_model.trainable_variables)]
        critic_params = [local_params.assign(global_params) for local_params, global_params in 
                            zip(self.critic_model.trainable_variables, global_network.critic_model.trainable_variables)]
        return actor_params, critic_params


    @tf.function
    def choose_action(self, s):  # run by a local
        prob_weights = self.actor_model.predict(s[np.newaxis, :])
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

class Worker(object):
    def __init__(self, name, global_network):
        self.env = MobiEnvironment(N_BS, N_UE, AREA_W)#gym.make(GAME).unwrapped
        self.name = name
        self.global_network = global_network
        self.AC = ACNet(name)
        self.total_steps = 0
        self.buf_r_dissect_all_ep = []
        self.step_start = 0
        self.step_end = 0

    
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        
        buffer_s, buffer_a, buffer_r = [], [], []
        print("worker ", self.name, "starts training")
        
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            #             s = self.env.reset()
            s = np.ravel(self.env.reset())
            ep_r = 0
            buf_r_dissect = []
            
            while True:
                a = self.AC.choose_action(s)
                
#                self.step_start = time.time()
                s_, r, done, info = self.env.step(a)
#                self.step_end = time.time()
#                print self.name," (env) step time ", self.step_end - self.step_start

                s_ = np.ravel(s_)
                
                ep_r += r
#                step_r = r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                
                if self.name == 'W_0': buf_r_dissect.append(info[0])
                
                if self.total_steps % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if self.total_steps % (UPDATE_GLOBAL_ITER*50) == 0:
                        print(self.name, "updating GlobalAC at step ", self.total_steps)
                    
#                    self.update_start= time.time()
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.critic_model, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    
                    buffer_v_target.reverse()
                    
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_from_global(self.global_network)
#                    self.update_end= time.time()
#                    print self.name," (agent) update time ", self.update_end - self.update_start

                s = s_

                self.total_steps += 1
                
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                          self.name,
                          "Ep:", GLOBAL_EP,
                          "| Ep_r: %f" % GLOBAL_RUNNING_R[-1],
                          "| total steps", self.total_steps,
                          "| step in Ep ", info[1]
                    )
              
                    GLOBAL_EP += 1

                    if self.name == 'W_0':
                        self.buf_r_dissect_all_ep.append(buf_r_dissect)
                        np.save("train/Reward_dissect", self.buf_r_dissect_all_ep)

                    if GLOBAL_EP % 500 == 0:
                        np.savez("train/Global_A_PARA" + str(GLOBAL_EP), SESS.run(GLOBAL_AC.a_params))

                    np.save("train/Global_return",GLOBAL_RUNNING_R)
#                    np.savez("train/A_PARA",SESS.run(self.AC.a_params))
                    np.savez("train/Global_A_PARA",SESS.run(GLOBAL_AC.a_params))

                    break


if __name__ == "__main__":
    print(">>>>>>>>>>>>>>>>A3C SIM INFO>>>>>>>>>>>>>>>>>>>>")
    print("tensor seed: ", TENSOR_SEED)
    print("N_S", N_S)
    print("N_A", N_A)
    print("LR_C", LR_C)
    print("N_BS", N_BS)
    print("N_UE", N_UE)
    print("AREA_W", AREA_W)
    print("Num of episodes", MAX_GLOBAL_EP)
    print("(if cnn), num of filters", CNN_NUM_FILTERS)
    print("(if cnn), num of filters", CNN_KERNEL_SIZE)
    print(">>>>>>>>>>>>>>>>>>>>SIM INFO(end)>>>>>>>>>>>>>>>")
    
    SESS = tf.Session()

    start = time.time()
    
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params


        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker namei
            print("Creating worker ", i_name)
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    np.savez("train/Global_A_PARA_init", SESS.run(GLOBAL_AC.a_params))

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
    print("Total time ", (end - start))

