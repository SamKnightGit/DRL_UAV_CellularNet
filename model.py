import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, initializers
from collections import deque



class A3CNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space, value_weight=0.5, entropy_coefficient=0.01):
        super(A3CNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.value_weight = value_weight
        self.entropy_coefficient = entropy_coefficient

        self.dense_shared = layers.Dense(1000, activation='relu', input_dim=self.state_space,
                                                kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden2 = layers.Dense(500, activation='relu',
        #                                         kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden3 = layers.Dense(250, activation='relu',
        #                                         kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden4 = layers.Dense(100, activation='relu',
        #                                         kernel_initializer=initializers.glorot_uniform)

        # self.dense_critic_hidden1 = layers.Dense(1000, activation='relu', input_dim=self.state_space,
        #                                          kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden2 = layers.Dense(500, activation='relu',
        #                                          kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden3 = layers.Dense(250, activation='relu',
        #                                          kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden4 = layers.Dense(100, activation='relu',
        #                                          kernel_initializer=initializers.glorot_uniform)

        self.policy = layers.Dense(self.action_space, activation=tf.nn.softmax,
                                   kernel_initializer=initializers.glorot_uniform)
        self.value = layers.Dense(1, kernel_initializer=initializers.glorot_uniform)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        policy_output = self.dense_shared(inputs)
        # policy_output = self.dense_critic_hidden2(policy_output)
        # policy_output = self.dense_critic_hidden3(policy_output)
        # policy_output = self.dense_critic_hidden4(policy_output)
        policy = self.policy(policy_output)

        value_output = self.dense_shared(inputs)
        # value_output = self.dense_value_hidden2(value_output)
        # value_output = self.dense_value_hidden3(value_output)
        # value_output = self.dense_value_hidden4(value_output)
        value = self.value(value_output)

        return policy, value

    def get_loss(self, done, new_state, history, discount_factor=0.99):
        if done:
            estimated_reward = 0
        else:
            _, estimated_reward = self(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))

        all_discounted_rewards = deque()
        for reward in history.rewards[::-1]:
            discounted_reward = estimated_reward * discount_factor
            discounted_reward += reward
            all_discounted_rewards.appendleft(discounted_reward)

        action_one_hot = tf.one_hot(history.actions, self.action_space, 1.0, 0.0)

        policy, values = self(tf.convert_to_tensor(np.vstack(history.states), dtype=tf.float32))

        advantage = tf.convert_to_tensor(np.array(all_discounted_rewards)[:, None], dtype=tf.float32) - values

        value_loss = tf.reduce_mean(tf.square(advantage))

        log_policy = tf.math.log(tf.clip_by_value(policy, 0.000001, 0.999999))
        log_policy_given_action = tf.reduce_sum(tf.multiply(log_policy, action_one_hot))

        policy_loss = -tf.reduce_mean(log_policy_given_action * advantage)

        entropy = tf.reduce_sum(tf.multiply(policy, -log_policy))

        total_loss = policy_loss + self.value_weight * value_loss - entropy * self.entropy_coefficient
        return total_loss


class ADQNetwork(tf.keras.Model):
    def __init__(self, state_space, action_space):
        super(ADQNetwork, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.dense_shared_1 = layers.Dense(400, activation='relu', input_dim=self.state_space,
                                           kernel_initializer=initializers.glorot_uniform)
        self.dense_shared_2 = layers.Dense(400, activation='relu', input_dim=self.state_space,
                                           kernel_initializer=initializers.glorot_uniform)
        # self.dense_value_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                        kernel_initializer=initializers.glorot_uniform)
        # self.dense_critic_hidden = layers.Dense(100, activation='relu', input_dim=self.state_space,
        #                                         kernel_initializer=initializers.glorot_uniform)
        self.q_value = layers.Dense(self.action_space, activation=tf.nn.softmax,
                                    kernel_initializer=initializers.glorot_uniform)
        # Initialize network weights with random input
        self(tf.convert_to_tensor(np.random.random((1, self.state_space)), dtype=tf.float32))

    def call(self, inputs):
        shared_output = self.dense_shared_1(inputs)
        shared_output = self.dense_shared_2(shared_output)

        q_value = self.q_value(shared_output)

        return q_value

    def get_loss(self, history):
        total_loss = 0
        for history_index in range(len(history.targets)):
            target = history.targets[history_index]
            state = history.states[history_index]
            action = history.actions[history_index]
            state = np.expand_dims(state, axis=0)
            action_prob = self(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            total_loss += target - tf.squeeze(action_prob)[action]
        return total_loss








