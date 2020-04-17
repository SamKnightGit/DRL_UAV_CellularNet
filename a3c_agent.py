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


class Worker(threading.Thread):
    global_episode = 0
    global_average_running_reward = 0
    best_score = 0
    best_checkpoint_score = 0
    save_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    update_lock = threading.Lock()
    best_worker_index = 0

    def __init__(self,
                 worker_index,
                 global_network,
                 num_base_stations,
                 num_users,
                 arena_width,
                 random_seed,
                 max_episodes,
                 optimizer,
                 update_frequency,
                 entropy_coefficient,
                 norm_clip_value,
                 num_checkpoints,
                 reward_queue,
                 save_dir):
        super(Worker, self).__init__()
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        self.env = MobiEnvironment(num_base_stations, num_users, arena_width)
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.update_frequency = update_frequency,
        self.norm_clip_value = norm_clip_value
        self.episodes_per_checkpoint = int(max_episodes / num_checkpoints)
        self.reward_queue = reward_queue
        self.save_dir = save_dir
        for checkpoint in range(num_checkpoints):
            os.makedirs(os.path.join(save_dir, f"checkpoint_{checkpoint}"), exist_ok=True)
        self.global_network = global_network
        self.local_network = model.A3CNetwork(self.state_space, self.action_space, entropy_coefficient=entropy_coefficient)

        self.reward_breakdown = []
        self.base_station_locations = []
        self.actions = []
        self.base_station_actions = []
        self.user_locations = []
        self.outage_fraction = []

    def _save_global_weights(self, filename):
        self.global_network.save_weights(filename)

    def _get_next_episode(self):
        with Worker.checkpoint_lock:
            episode = Worker.global_episode
            Worker.global_episode += 1
        return episode

    def _clear_info(self):
        self.reward_breakdown = []
        self.base_station_locations = []
        self.actions = []
        self.base_station_actions = []
        self.user_locations = []
        self.outage_fraction = []

    def _record_initial_info(self):
        # self.base_station_locations.append(copy.deepcopy(self.env.bsLoc))
        # self.user_locations.append(copy.deepcopy(self.env.ueLoc))
        pass

    def _record_info(self, info, real_action):
        # self.reward_breakdown.append(info.r_dissect)
        # self.base_station_locations.append(info.bs_loc)
        # self.actions.append(real_action)
        # self.base_station_actions.append(info.bs_actions)
        # self.user_locations.append(info.ue_loc)
        # self.outage_fraction.append(info.outage_fraction)
        pass

    def _save_info(self, checkpoint, is_best=False):
        if is_best:
            worker_name = "worker_best"
        else:
            worker_name = "worker_0"
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_{checkpoint}")
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_reward_breakdown"), self.reward_breakdown)
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_base_station_locations"), self.base_station_locations)
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_base_station_actions"), self.base_station_actions)
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_instructed_actions"), self.actions)
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_user_locations"), self.user_locations)
        np.save(os.path.join(checkpoint_dir, f"{worker_name}_outage_fraction"), self.outage_fraction)

    def run(self):
        history = History()
        update_counter = 0
        ep_reward = 0
        global_episode = self._get_next_episode()
        while global_episode < self.max_episodes:
            print(f"Starting global episode: {global_episode}")
            current_state = np.ravel(self.env.reset())
            history.clear()
            self._record_initial_info()
            # print(self.env.bsLoc)

            done = False
            while not done:
                action_prob, _ = self.local_network(
                    tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
                )
                action_prob = tf.squeeze(action_prob).numpy()
                action = np.random.choice(self.action_space, p=action_prob)
                new_state, reward, done, info = self.env.step(action)
                # print(info.bs_loc)
                new_state = np.ravel(new_state)
                if done:
                    reward = -1
                ep_reward += reward

                history.append(current_state, action, reward)

                self._record_info(info, action)

                if update_counter == self.update_frequency or done:
                    with Worker.update_lock:
                        with tf.GradientTape() as tape:
                            local_loss = self.local_network.get_loss(done, new_state, history)
                        local_gradients = tape.gradient(local_loss, self.local_network.trainable_weights)
                        if self.norm_clip_value:
                            local_gradients, _ = tf.clip_by_global_norm(local_gradients, self.norm_clip_value)
                        self.optimizer.apply_gradients(
                            zip(local_gradients, self.global_network.trainable_weights)
                        )
                        self.local_network.set_weights(self.global_network.get_weights())

                    history.clear()
                    update_counter = 0

                update_counter += 1
                current_state = new_state

            current_checkpoint = int(global_episode / self.episodes_per_checkpoint)
            checkpoint_model_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}", "model.h5")
            global_model_path = os.path.join(self.save_dir, "best_model.h5")
            with Worker.save_lock:
                if ep_reward >= Worker.best_checkpoint_score:
                    if ep_reward >= Worker.best_score:
                        print(f"New global best score of {ep_reward} achieved by Worker {self.name}!")
                        self.global_network.save_weights(global_model_path)
                        print(
                            "Saved global best model at: " +
                            f"{os.path.join(self.save_dir, 'checkpoint_best.h5')}"
                        )
                        self._save_info(current_checkpoint, is_best=True)
                        Worker.best_score = ep_reward
                    print(f"New checkpoint best score of {ep_reward} achieved by Worker {self.name}!")
                    self._save_global_weights(checkpoint_model_path)
                    print(f"Saved checkpoint best model at: {checkpoint_model_path}")
                    Worker.best_checkpoint_score = ep_reward
                    Worker.best_worker_index = self.worker_index

                if not os.path.exists(checkpoint_model_path):
                    self._save_global_weights(checkpoint_model_path)
                    Worker.best_checkpoint_score = 0

                if Worker.global_average_running_reward == 0:
                    Worker.global_average_running_reward = ep_reward
                else:
                    Worker.global_average_running_reward = Worker.global_average_running_reward * 0.99 + ep_reward * 0.01
                self.reward_queue.put(Worker.global_average_running_reward)
            global_episode = self._get_next_episode()
            ep_reward = 0

            checkpoint_np_file = os.path.join(
                self.save_dir,
                f"checkpoint_{current_checkpoint}",
                "reward_breakdown.npy"
            )
            if self.worker_index == 0 and not os.path.exists(checkpoint_np_file):
                self._save_info(current_checkpoint)
            self._clear_info()
        self.reward_queue.put(None)


class TestWorker(threading.Thread):
    def __init__(self,
                 global_network,
                 num_base_stations,
                 num_users,
                 arena_width,
                 max_episodes,
                 test_file_name,
                 render=True,
                 random_seed=None):
        super(TestWorker, self).__init__()
        self.global_network = global_network
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
        self.base_station_locations = []
        self.actions = []
        self.base_station_actions = []
        self.user_locations = []
        

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
                action_prob, _ = self.global_network(
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

