import threading
import tensorflow as tf
import model
import numpy as np
import os
import copy
import time
from mobile_env_original import MobiEnvironment


class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Coordinator:
    def __init__(self, 
                 network, 
                 num_workers, 
                 num_base_stations,
                 num_users,
                 arena_width,
                 cutoff_sinr,
                 timesteps_per_rollout, 
                 timesteps_per_episode, 
                 num_episodes,
                 num_checkpoints,
                 norm_clip_value,
                 optimizer,
                 random_seed,
                 save_dir,
                 summary_writer):
        self.network = network
        self.workers = []
        for worker_index in range(num_workers):
            worker = Worker(
                worker_index, 
                num_base_stations,
                num_users,
                arena_width,
                cutoff_sinr,
                network,
                timesteps_per_rollout,
                random_seed
            )
            self.workers.append(worker)
        self.timesteps_per_rollout = timesteps_per_rollout
        self.timesteps_per_episode = timesteps_per_episode
        self.num_episodes = num_episodes
        self.episodes_per_checkpoint = int(num_episodes / num_checkpoints)
        self.norm_clip_value = norm_clip_value
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.summary_writer = summary_writer
        self.smoothed_reward = []

    def add_to_smoothed_reward(self, reward):
        if len(self.smoothed_reward) == 0:
            # existing_reward = 0
            self.smoothed_reward.append(reward)
        else:
            # existing_reward = self.smoothed_reward[-1]
            self.smoothed_reward.append(0.01 * reward + 0.99 * self.smoothed_reward[-1])
        # self.smoothed_reward.append(0.01 * reward + 0.99 * existing_reward)
    
    def run(self):
        rollouts_per_episode = int(self.timesteps_per_episode / self.timesteps_per_rollout)
        current_checkpoint = 0
        best_checkpoint_reward = 0
        timestep = 0
        episode = 0
        while episode < self.num_episodes:
            next_checkpoint = int(episode / self.episodes_per_checkpoint)
            if (next_checkpoint != current_checkpoint):
                best_checkpoint_reward = 0
                current_checkpoint = next_checkpoint
            for worker in self.workers:
                if episode < self.num_episodes:
                    worker.episode = episode
                    worker.reset_env()
                    episode += 1
            for rollout in range(rollouts_per_episode):
                for worker in self.workers:
                    print(f"Processing in worker {worker.worker_index}, rollout {rollout} of episode {worker.episode}")
                    done, new_state, history = worker.work()
                    with tf.GradientTape() as tape:
                        loss = self.network.get_loss(done, new_state, history)
                        if worker.worker_index == 0:
                            with self.summary_writer.as_default():
                                tf.summary.scalar('loss', loss, timestep)
                    gradients = tape.gradient(loss, self.network.trainable_weights)
                    if self.norm_clip_value:
                        gradients, _ = tf.clip_by_global_norm(gradients, self.norm_clip_value)
                    self.optimizer.apply_gradients(
                        zip(gradients, self.network.trainable_weights)
                    )
                    if done:
                        print(f"Worker {worker.worker_index} finished. Resetting environment!")
                        if worker.ep_reward >= best_checkpoint_reward:
                            print(f"-------\n Best checkpoint score of {worker.ep_reward} achieved\n-------")
                            best_checkpoint_reward = worker.ep_reward
                        with self.summary_writer.as_default():
                            tf.summary.scalar('ep_reward', worker.ep_reward, worker.episode)
                        self.add_to_smoothed_reward(worker.ep_reward)
                    if not os.path.exists(os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")):
                        self.network.save_weights(
                            os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}.h5")
                        )
                    timestep += 1

class Worker:
    def __init__(self,
                 worker_index,
                 num_base_stations,
                 num_users,
                 arena_width,
                 cutoff_sinr,
                 network,
                 timesteps_per_rollout,
                 random_seed):
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        # self.env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
        self.env = MobiEnvironment(num_base_stations, num_users, arena_width)
        self.cutoff_sinr = cutoff_sinr
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.state = np.ravel(self.env.reset())
        self.network = network
        self.timesteps_per_rollout = timesteps_per_rollout
        self.ep_reward = 0
        self.done = False
        self.episode = -1

    def reset_env(self):
        self.state = np.ravel(self.env.reset())
        self.done = False
        self.ep_reward = 0

    def work(self):
        history = History()
        timestep = 0
        current_state = self.state

        done = False
        while timestep < self.timesteps_per_rollout:
            action_prob, _ = self.network(
                tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
            )
            action_prob = tf.squeeze(action_prob).numpy()

            action = np.random.choice(self.action_space, p=action_prob)
            new_state, reward, done, _ = self.env.step(action, cutoff_sinr=self.cutoff_sinr)

            new_state = np.ravel(new_state)
            if done:
                reward = -1
            self.ep_reward += reward
            history.append(current_state, action, reward)
            current_state = np.ravel(new_state)
            timestep += 1
        self.done = done
        return done, current_state, history



class TestWorker:
    def __init__(self,
                 num_base_stations,
                 num_users,
                 arena_width,
                 network,
                 max_episodes,
                 test_file_name,
                 render=True,
                 random_seed=None):
        super(TestWorker, self).__init__()
        self.network = network
        # self.env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
        self.env = MobiEnvironment(num_base_stations, num_users, arena_width, "read_trace", "./ue_trace_10k.npy")
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.max_episodes = max_episodes
        self.test_file_name = test_file_name
        self.render = render

        self.reward_breakdown = []
        self.sinr_all = []
        self.time_all = []
        self.outage_all = []
        # self.base_station_locations = []
        # self.actions = []
        # self.base_station_actions = []
        # self.user_locations = []
        # self.outage_fraction = []

    def _record_initial_info(self):
        # self.base_station_locations.append(copy.deepcopy(self.env.bsLoc))
        # self.user_locations.append(copy.deepcopy(self.env.ueLoc))
        pass

    def _record_info(self, info, start_time, real_action=None):
        # self.reward_breakdown.append(info.r_dissect)
        # self.base_station_locations.append(info.bs_loc)
        # self.actions.append(real_action)
        # self.base_station_actions.append(info.bs_actions)
        # self.user_locations.append(info.ue_loc)
        # self.outage_fraction.append(info.outage_fraction)
        self.reward_breakdown.append(info[0])
        self.sinr_all.append(self.env.channel.current_BS_sinr)
        self.time_all.append(time.time() - start_time)
        self.outage_all.append(info[3])

    def _save_info(self):
        test_dir = os.path.dirname(self.test_file_name)
        # np.save(os.path.join(test_dir, "reward_breakdown"), self.reward_breakdown)
        # np.save(os.path.join(test_dir, "base_station_locations"), self.base_station_locations)
        # np.save(os.path.join(test_dir, "base_station_actions"), self.base_station_actions)
        # np.save(os.path.join(test_dir, "instructed_actions"), self.actions)
        # np.save(os.path.join(test_dir, "user_locations"), self.user_locations)
        # np.save(os.path.join(test_dir, "outage_fraction"), self.outage_fraction)
        np.save(os.path.join(test_dir, "reward"), self.reward_breakdown)
        np.save(os.path.join(test_dir, "sinr"), self.sinr_all)
        np.save(os.path.join(test_dir, "time"), self.time_all)
        np.save(os.path.join(test_dir, "outage"), self.outage_all)

    def run(self):
        episode = 0
        best_reward = 0
        average_reward = 0
        while episode < self.max_episodes:
            current_state = np.ravel(self.env.reset())
            self._record_initial_info()

            ep_reward = 0

            done = False
            while not done:
                if self.render:
                    self.env.render()
                action_prob, _ = self.network(
                    tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
                )
                action_prob = tf.squeeze(action_prob).numpy()
                action = np.argmax(action_prob)

                new_state, reward, done, info = self.env.step_test(action)
                self._record_info(info, action)
                new_state = np.ravel(new_state)

                if done:
                    reward = -1
                ep_reward += reward

                current_state = new_state
            if ep_reward > best_reward:
                best_reward = ep_reward
            average_reward += ep_reward
            episode += 1
        average_reward /= self.max_episodes
        if self.test_file_name:
            with open(self.test_file_name, "w+") as fp:
                fp.write("Best Reward:".ljust(20) + f"{best_reward}\n")
                fp.write("Average Reward:".ljust(20) + f"{average_reward}\n")
        self._save_info()

