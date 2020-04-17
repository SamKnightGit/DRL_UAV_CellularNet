import threading
import time
import tensorflow as tf
import model
import numpy as np
import os
import copy
from mobile_env_original import MobiEnvironment


class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.targets = []

    def append(self, state, action, target):
        self.states.append(state)
        self.actions.append(action)
        self.targets.append(target)

    def clear(self):
        self.states = []
        self.actions = []
        self.targets = []


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
                 num_base_stations,
                 num_users,
                 arena_width,
                 cutoff_sinr,
                 random_seed,
                 max_episodes,
                 optimizer,
                 target_network,
                 main_network,
                 target_update_frequency,
                 update_frequency,
                 min_epsilon,
                 epsilon_annealing_strategy,
                 annealing_episodes,
                 discount_factor,
                 norm_clip_value,
                 clipped_reward,
                 num_checkpoints,
                 reward_queue,
                 logging_frequency,
                 summary_writer,
                 save_dir,
                 save=True):
        super(Worker, self).__init__()
        self.worker_index = worker_index
        self.name = f"Worker_{worker_index}"
        # self.env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
        self.env = MobiEnvironment(num_base_stations, num_users, arena_width)
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.cutoff_sinr = cutoff_sinr
        self.max_episodes = max_episodes
        self.optimizer = optimizer
        self.target_network = target_network
        self.main_network = main_network
        self.target_update_frequency = target_update_frequency
        self.update_frequency = update_frequency
        self.min_epsilon = min_epsilon
        self.start_epsilon = 1.0
        self.epsilon = self.start_epsilon
        self.annealing_episodes = annealing_episodes
        self.epsilon_anneal_quantity = self._calculate_epsilon_anneal(epsilon_annealing_strategy)
        self.discount_factor = discount_factor
        self.norm_clip_value = norm_clip_value
        self.clipped_reward = clipped_reward
        self.episodes_per_checkpoint = int(max_episodes / num_checkpoints)
        self.reward_queue = reward_queue
        self.logging_frequency = logging_frequency
        self.summary_writer = summary_writer
        self.save_dir = save_dir
        self.save = save
        if save:
            for checkpoint in range(num_checkpoints):
                os.makedirs(os.path.join(save_dir, f"checkpoint_{checkpoint}"), exist_ok=True)

        self.gradients = []
        self.reward_breakdown = []
        self.base_station_locations = []
        self.actions = []
        self.base_station_actions = []
        self.user_locations = []
        self.outage_fraction = []

    def _save_gradients(self, file_path):
        if self.save:
            np.save(file_path, self.gradients)
            self.gradients = []

    def _save_global_weights(self, model_path):
        if self.save:
            self.main_network.save_weights(model_path)

    def _calculate_target(self, new_state, reward, done):
        if done:
            target_output = reward
        else:
            target_action_values = self.target_network(
                tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)
            )
            target_action_values = tf.squeeze(target_action_values)
            #DQN Update
            #greedy_action_value = np.max(tf.squeeze(target_action_prob).numpy())
            #Double DQN Update
            main_action_values = self.main_network(
                tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)
            )
            main_action_values = tf.squeeze(main_action_values)
            decoupled_action_value = target_action_values[tf.math.argmax(main_action_values)]
            target_output = reward + self.discount_factor * decoupled_action_value
        return target_output

    def _calculate_epsilon_anneal(self, strategy):
        if strategy == "linear":
            return self.epsilon / self.annealing_episodes
        else:
            raise NotImplementedError
    
    def _anneal_epsilon(self, episode):
        next_epsilon = self.start_epsilon - (episode * self.epsilon_anneal_quantity)
        if next_epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon
        else:
            self.epsilon = next_epsilon

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
        self.base_station_locations.append(copy.deepcopy(self.env.bsLoc))
        self.user_locations.append(copy.deepcopy(self.env.ueLoc))

    def _record_info(self, info, real_action):
        self.reward_breakdown.append(info.r_dissect)
        self.base_station_locations.append(info.bs_loc)
        self.actions.append(real_action)
        self.base_station_actions.append(info.bs_actions)
        self.user_locations.append(info.ue_loc)
        self.outage_fraction.append(info.outage_fraction)

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
        timestep = 0
        ep_reward = 0
        global_episode = self._get_next_episode()
        while global_episode < self.max_episodes:
            print(f"Starting global episode: {global_episode}")
            print(f"   with epsilon value {self.epsilon}")
            if global_episode != 0 and global_episode % self.target_update_frequency == 0:
                print("Updating target network!")
                self.target_network.set_weights(self.main_network.get_weights())
            current_state = np.ravel(self.env.reset())
            history.clear()
            # self._record_initial_info()
            self.gradients = []

            done = False
            while not done:
                if timestep % 100 == 0:
                    print(f"In Worker {self.worker_index} at timestep {timestep}")
                if self.epsilon > np.random.random():
                    action = np.random.choice(self.action_space)
                else:
                    action_values = self.main_network(
                        tf.convert_to_tensor(current_state[np.newaxis, :], dtype=tf.float32)
                    )
                    action_values = tf.squeeze(action_values).numpy()
                    action = np.argmax(action_values)
                    if self.worker_index == 0 and timestep % self.logging_frequency == 0:
                        with self.summary_writer.as_default():
                            tf.summary.scalar('max Q', action_values[action], timestep)

                new_state, reward, done, info = self.env.step(action, cutoff_sinr=self.cutoff_sinr, clipped_reward=self.clipped_reward)
                new_state = np.ravel(new_state)
                if done:
                    reward = -1
                ep_reward += reward

                history.append(current_state, action, self._calculate_target(new_state, reward, done))

                # self._record_info(info, action)

                if update_counter == self.update_frequency or done:
                    with Worker.update_lock:
                        with tf.GradientTape() as tape:
                            local_loss = self.main_network.get_loss(history)
                        if self.worker_index == 0 and timestep % self.logging_frequency == 0:
                            with self.summary_writer.as_default():
                                tf.summary.scalar('loss', local_loss, timestep)
                        local_gradients = tape.gradient(local_loss, self.main_network.trainable_weights)
                        if self.norm_clip_value:
                            local_gradients, _ = tf.clip_by_global_norm(local_gradients, self.norm_clip_value)
                        if self.worker_index == 0:
                            pass
                            # self.gradients.append(local_gradients)
                        self.optimizer.apply_gradients(
                            zip(local_gradients, self.main_network.trainable_weights)
                        )
                    history.clear()
                    update_counter = 0
                timestep += 1
                update_counter += 1
                current_state = new_state

            with self.summary_writer.as_default():
                tf.summary.scalar('ep_reward', ep_reward, global_episode)

            current_checkpoint = int(global_episode / self.episodes_per_checkpoint)
            checkpoint_model_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}", "model.h5")
            global_model_path = os.path.join(self.save_dir, "best_model.h5")
            gradient_path = os.path.join(self.save_dir, f"checkpoint_{current_checkpoint}", "gradients")

            with Worker.save_lock:
                if ep_reward >= Worker.best_checkpoint_score:
                    if ep_reward >= Worker.best_score:
                        print(f"New global best score of {ep_reward} achieved by Worker {self.name}!")
                        self._save_global_weights(global_model_path)
                        print(f"Saved global best model at: {os.path.join(self.save_dir, 'checkpoint_best.h5')}")
                        print(f"Model Loss: {local_loss}")
                        # self._save_info(current_checkpoint, is_best=True)
                        Worker.best_score = ep_reward
                    print(f"New checkpoint best score of {ep_reward} achieved by Worker {self.name}!")
                    self._save_global_weights(checkpoint_model_path)
                    print(f"Saved checkpoint best model at: {checkpoint_model_path}")
                    Worker.best_checkpoint_score = ep_reward
                    Worker.best_worker_index = self.worker_index

                if not os.path.exists(checkpoint_model_path):
                    self._save_global_weights(checkpoint_model_path)
                    Worker.best_checkpoint_score = 0

                if self.worker_index == 0 and (not os.path.exists(gradient_path)):
                    pass
                    # self._save_gradients(gradient_path)
                if Worker.global_average_running_reward == 0:
                    Worker.global_average_running_reward = ep_reward
                else:
                    Worker.global_average_running_reward = Worker.global_average_running_reward * 0.99 + ep_reward * 0.01
                self.reward_queue.put(Worker.global_average_running_reward)
            self._anneal_epsilon(global_episode)
            global_episode = self._get_next_episode()
            ep_reward = 0

            checkpoint_np_file = os.path.join(
                self.save_dir,
                f"checkpoint_{current_checkpoint}",
                "reward_breakdown.npy"
            )
            if self.worker_index == 0 and not os.path.exists(checkpoint_np_file):
                # self._save_info(current_checkpoint)  
                pass          
            # self._clear_info()
        self.reward_queue.put(None)


class TestWorker(threading.Thread):
    def __init__(self,
                 global_network,
                 num_base_stations,
                 num_users,
                 arena_width,
                 cutoff_sinr,
                 max_episodes,
                 test_file_name,
                 render=True,
                 random_seed=None):
        super(TestWorker, self).__init__()
        self.global_network = global_network
        # self.env = MobiEnvironment(num_base_stations, num_users, arena_width, random_seed=random_seed)
        self.env = MobiEnvironment(num_base_stations, num_users, 100, "read_trace", "./ue_trace_10k.npy")
        self.state_space = self.env.observation_space_dim
        self.action_space = self.env.action_space_dim
        self.cutoff_sinr = cutoff_sinr
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
        self.outage_fraction = []
        self.sinr_all = []
        self.time_all = []

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
                action_prob = self.global_network(
                    tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                )
                action_prob = tf.squeeze(action_prob).numpy()
                action = np.argmax(action_prob)
                new_state, reward, done, info = self.env.step_test(action, cutoff_sinr=self.cutoff_sinr)
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
                fp.write(f"Best Reward: {best_reward}\n")
                fp.write(f"Average Reward: {average_reward}\n")
        self._save_info()