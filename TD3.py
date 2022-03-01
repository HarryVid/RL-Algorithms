__author__ = "Hari Vidharth"
__license__ = "Open Source"
__version__ = "1.0"
__maintainer__ = "Hari Vidharth"
__email__ = "viju1145@gmail.com"
__date__ = "Feb 2022"
__status__ = "Prototype"


import gc
import time

import gym
import panda_gym
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow_addons.layers import NoisyDense
from tensorflow_addons.optimizers import Lookahead
from tensorflow.keras import backend


gc.collect()
backend.clear_session()


class ExperienceReplay():
	def __init__(self, size, state_shape, action_shape):
		self.size = size
		self.memory_counter = 0

		self.state_memory = np.zeros((self.size, *state_shape))
		self.action_memory = np.zeros((self.size, action_shape))
		self.reward_memory = np.zeros(self.size)
		self.next_state_memory = np.zeros((self.size, *state_shape))
		self.terminal_memory = np.zeros(self.size, dtype=bool)

	def save_to_memory(self, state, action, reward, next_state, done):
		index = self.memory_counter % self.size

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminal_memory[index] = done

		self.memory_counter += 1

	def sample_from_memory(self, batch_size):
		sample_size = min(self.memory_counter, self.size) - 1
		batch = np.random.choice(sample_size, batch_size - 1, replace=False)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		next_states = self.next_state_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, next_states, dones


class CombinedExperienceReplay():
	def __init__(self, size, state_shape, action_shape):
		self.size = size
		self.memory_counter = 0

		self.state_memory = np.zeros((self.size, *state_shape))
		self.action_memory = np.zeros((self.size, action_shape))
		self.reward_memory = np.zeros(self.size)
		self.next_state_memory = np.zeros((self.size, *state_shape))
		self.terminal_memory = np.zeros(self.size, dtype=bool)

	def save_to_memory(self, state, action, reward, next_state, done):
		index = self.memory_counter % self.size

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminal_memory[index] = done

		self.memory_counter += 1

	def sample_from_memory(self, batch_size):
		sample_size = min(self.memory_counter, self.size) - 1
		batch = np.random.choice(sample_size, batch_size - 1, replace=False)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		next_states = self.next_state_memory[batch]
		dones = self.terminal_memory[batch]

		index = self.memory_counter % self.size - 1

		last_state = self.state_memory[index]
		last_action = self.action_memory[index]
		last_reward = self.reward_memory[index]
		last_next_state = self.next_state_memory[index]
		last_done = self.terminal_memory[index]

		c_states = np.vstack((states, last_state))
		c_actions = np.vstack((actions, last_action))
		c_rewards = np.append(rewards, last_reward)
		c_next_states = np.vstack((next_states, last_next_state))
		c_dones = np.append(dones, last_done)

		return c_states, c_actions, c_rewards, c_next_states, c_dones


class CriticNetwork(Model):
	def __init__(self):
		super(CriticNetwork, self).__init__()

		w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		self.fc1 = Dense(64, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc2 = Dense(64, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc3 = Dense(64, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.q = Dense(1, activation="linear", kernel_initializer=w_init, kernel_regularizer="l2")

	def call(self, state, action):
		x = self.fc1(tf.concat([state, action], axis=1))
		x = self.fc2(x)
		x = self.fc3(x)
		q = self.q(x)

		return q


class ActorNetwork(Model):
	def __init__(self, action_shape):
		super(ActorNetwork, self).__init__()

		w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

		self.fc1 = NoisyDense(64, sigma=0.1, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc2 = NoisyDense(64, sigma=0.1, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.fc3 = NoisyDense(64, sigma=0.1, activation="elu", kernel_initializer="he_uniform", kernel_regularizer="l2")
		self.a = NoisyDense(action_shape, sigma=0.1, activation="tanh", kernel_initializer=w_init, kernel_regularizer="l2")

	def call(self, state):
		x = self.fc1(state)
		x = self.fc2(x)
		x = self.fc3(x)
		a = self.a(x)

		return a


class Agent():
	def __init__(self, env):
		self.env = env
		self.learn_counter = 0

        # self.agent_memory = ExperienceReplay(1000000, (9, ), env.action_space.shape[0])
		self.agent_memory = CombinedExperienceReplay(1000000, (9, ), env.action_space.shape[0])

		self.actor = ActorNetwork(env.action_space.shape[0])
		self.critic_1 = CriticNetwork()
		self.critic_2 = CriticNetwork()
		self.target_actor = ActorNetwork(env.action_space.shape[0])
		self.target_critic_1 = CriticNetwork()
		self.target_critic_2 = CriticNetwork()

		self.actor.compile(optimizer=Lookahead("adam"))
		self.critic_1.compile(optimizer=Lookahead("adam"))
		self.critic_2.compile(optimizer=Lookahead("adam"))
		self.target_actor.compile(optimizer=Lookahead("adam"))
		self.target_critic_1.compile(optimizer=Lookahead("adam"))
		self.target_critic_2.compile(optimizer=Lookahead("adam"))

		self.update_network_parameters(tau=1)

	def train_noisy_action(self, state):
        action_choice = np.random.choice([1, 2], p=[0.9, 0.1])
        
        if action_choice == 1:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            action = self.actor(state)[0]
            action += np.random.normal(scale=0.1)
            action = tf.clip_by_value(action, self.env.action_space.low[0], self.env.action_space.high[0])
        elif action_choice == 2:
				action = env.action_space.sample()

		return action

    def train_epsilon_greedy_action(self, state):
        pass

	def test_action(self, state):
		state = tf.convert_to_tensor([state], dtype=tf.float32)
		action = self.actor(state)[0]
		action = tf.clip_by_value(action, self.env.action_space.low[0], self.env.action_space.high[0])

		return action

	def agent_remember(self, state, action, reward, next_state, done):
		self.agent_memory.save_to_memory(state, action, reward, next_state, done)

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = 0.005

		weights = []
		targets = self.target_actor.weights
		for i, weight in enumerate(self.actor.weights):
			weights.append(weight * tau + targets[i] * (1 - tau))
		self.target_actor.set_weights(weights)

		weights = []
		targets = self.target_critic_1.weights
		for i, weight in enumerate(self.critic_1.weights):
			weights.append(weight * tau + targets[i] * (1 - tau))
		self.target_critic_1.set_weights(weights)

		weights = []
		targets = self.target_critic_2.weights
		for i, weight in enumerate(self.critic_2.weights):
			weights.append(weight * tau + targets[i] * (1 - tau))
		self.target_critic_2.set_weights(weights)

	def save_networks(self):
		print(" ")
		print("... Saving Networks! ...")
		print(" ")

		self.actor.save_weights("./checkpoints/Actor")
		self.critic_1.save_weights("./checkpoints/Critic1")
		self.critic_2.save_weights("./checkpoints/Critic2")
		self.target_actor.save_weights("./checkpoints/TargetActor")
		self.target_critic_1.save_weights("./checkpoints/TargetCritic1")
		self.target_critic_2.save_weights("./checkpoints/TargetCritic2")

	def load_networks(self):
		print(" ")
		print('... loading networks ...')
		print(" ")

		self.actor.load_weights("./checkpoints/Actor").expect_partial()
		self.critic_1.load_weights("./checkpoints/Critic1").expect_partial()
		self.critic_2.load_weights("./checkpoints/Critic2").expect_partial()
		self.target_actor.load_weights("./checkpoints/TargetActor").expect_partial()
		self.target_critic_1.load_weights("./checkpoints/TargetCritic1").expect_partial()
		self.target_critic_2.load_weights("./checkpoints/TargetCritic2").expect_partial()

	def learn(self):
		if self.agent_memory.memory_counter >= 512:
			states, actions, rewards, next_states, dones = self.agent_memory.sample_from_memory(512)

			states = tf.convert_to_tensor(states, dtype=tf.float32)
			actions = tf.convert_to_tensor(actions, dtype=tf.float32)
			rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
			next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

			target_actions = self.target_actor(next_states)
			target_actions += tf.clip_by_value(np.random.normal(scale=0.2), -50, 0.5)
			target_actions = tf.clip_by_value(target_actions, self.env.action_space.low[0], self.env.action_space.high[0])
			target_q1 = tf.squeeze(self.target_critic_1(next_states, target_actions), 1)
			target_q2 = tf.squeeze(self.target_critic_2(next_states, target_actions), 1)
			target_critic_value = tf.minimum(target_q1, target_q2)
			target_q = rewards + 0.98 * target_critic_value * (1 - dones)

			with tf.GradientTape() as critic_1_tape:
				critic_1_tape.watch(self.critic_1.trainable_variables)
				q1 = tf.squeeze(self.critic_1(states, actions), 1)
				critic_1_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target_q) - q1))
				critic_1_loss += tf.reduce_sum(self.critic_1.losses)
			critic_1_gradient = critic_1_tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
			self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
			del critic_1_tape

			with tf.GradientTape() as critic_2_tape:
				critic_2_tape.watch(self.critic_2.trainable_variables)
				q2 = tf.squeeze(self.critic_2(states, actions), 1)
				critic_2_loss = tf.reduce_mean(tf.square(tf.stop_gradient(target_q) - q2))
				critic_2_loss += tf.reduce_sum(self.critic_2.losses)
			critic_2_gradient = critic_2_tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
			self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))
			del critic_2_tape

			self.learn_counter += 1

			if self.learn_counter % 2 == 0:
				with tf.GradientTape() as actor_tape:
					actor_tape.watch(self.actor.trainable_variables)
					new_actions = self.actor(states)
					q1 = tf.squeeze(self.critic_1(states, new_actions), 1)
					q2 = tf.squeeze(self.critic_2(states, new_actions), 1)
					critic_value = tf.minimum(q1, q2)
					actor_loss = -tf.reduce_mean(critic_value)
					actor_loss += tf.reduce_sum(self.actor.losses)
				actor_gradient = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
				self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
				del actor_tape

				self.update_network_parameters()
