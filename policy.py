import random
import numpy as np

from env import Env

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from collections import deque

import time

import sys

class Policy:

	MIN_REPLAY_MEMORY_SIZE = 5000
	BATCH_SIZE = 64
	UPDATE_TARGET_EVERY = 5
	MODEL_NAME = "256_64"

	LEARNING_RATE = 0.4
	DISCOUNT = 0.85
	EPISODES = 400

	MIN_REWARD_INIT = -0.1
	MAX_REWARD_INIT = 0.1

	START_EPSILON_DECAYING = 1
	END_EPSILON_DECAYING = EPISODES // 2

	ARGMAX_EQ_TH = 0.2

	state_res_size = 0

	ex_states = []
	ex_actions = []
	ex_next_states = []

	ex_accuracies = []

	def __init__(self, env):

		self.env = env
		self.state_res_size = self.env.TOTAL_VISION * self.env.NB_GROUND_TYPES
		self.state_res_size += self.env.TOTAL_VISION * self.env.NB_OBJ_TYPES

		self.saved_action = 0
		self.saved_state = None

		self.epsilon = 1.0
		self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

		self.model = self.create_model()

		#self.target_model = self.create_model()
		#self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = []

		#self.target_update_counter = 0

	def create_model(self):

		model = Sequential()

		model.add(Dense(256, activation="linear", input_dim = (self.state_res_size + self.env.NB_MOVES)))
		model.add(Dense(256, activation="linear"))
		model.add(Dense(self.state_res_size, activation="sigmoid"))

		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def state_2_state_res(self, state):
		state_res = []

		for i in range(len(state)):
			if i % 2 == 0:#ground
				for j in range(self.env.NB_GROUND_TYPES):
					if state[i] == j:
						state_res.append(1)
					else:
						state_res.append(0)
			elif i % 2 == 1:#onground
				for j in range(self.env.NB_OBJ_TYPES):
					if state[i] == j:
						state_res.append(1)
					else:
						state_res.append(0)

		return state_res

	def state_res_2_state(self, res):
		state = []

		sub_res = []
		tmp = []
		cpt = 0
		for i in range(len(res)):
			if cpt < self.env.NB_GROUND_TYPES:
				tmp.append(res[i])
			elif cpt == self.env.NB_GROUND_TYPES:
				sub_res.append(tmp)
				tmp = []
				tmp.append(res[i])
			elif cpt < self.env.NB_GROUND_TYPES + self.env.NB_OBJ_TYPES:
				tmp.append(res[i])
			elif cpt == self.env.NB_GROUND_TYPES + self.env.NB_OBJ_TYPES:
				cpt = 0
				sub_res.append(tmp)
				tmp = []
				tmp.append(res[i])

			cpt += 1

		sub_res.append(tmp)

		for i in range(len(sub_res)):
			state.append(np.argmax(sub_res[i]))

		return state

	def update_replay_memory(self, over):
		transition = (self.saved_state, self.saved_action, self.env.get_state(), over)
		self.replay_memory.append(transition)

	def action_2_res(self, action):
		if action == self.env.MOVE_UP:
			return [1, 0, 0, 0]
		elif action == self.env.MOVE_DOWN:
			return [0, 1, 0, 0]
		elif action == self.env.MOVE_LEFT:
			return [0, 0, 1, 0]
		elif action == self.env.MOVE_RIGHT:
			return [0, 0, 0, 1]


	def train(self, terminal_state):
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		X = []
		y = []

		for transition in self.replay_memory:
			X_sample = self.state_2_state_res(transition[0])
			X_sample = np.concatenate((X_sample, self.action_2_res(transition[1])))
			X.append(X_sample)

			y.append(self.state_2_state_res(transition[2]))

		self.model.fit(np.array(X), np.array(y), batch_size=self.BATCH_SIZE, shuffle=False)

		self.save_some_examples()
		self.replay_memory = []

		#if terminal_state:
			#self.target_update_counter += 1

		#if self.target_update_counter > self.UPDATE_TARGET_EVERY:
			#self.target_model.set_weights(self.model.get_weights())
			#self.target_update_counter = 0

	def save_some_examples(self):
		for i in range(10):
			self.ex_states.append(self.replay_memory[i][0])
			self.ex_actions.append(self.replay_memory[i][1])
			self.ex_next_states.append(self.replay_memory[i][2])

	def test_model(self):

		for ex_id in range(len(self.ex_states)):

			x = self.state_2_state_res(self.ex_states[ex_id])
			x = np.concatenate((x, self.action_2_res(self.ex_actions[ex_id])))

			res = self.model.predict(np.array([x]))
			res_round = []
			for v in res[0]:
				res_round.append(int(round(v)))

			ex_res = self.state_2_state_res(self.ex_next_states[ex_id])

			errors = 0
			for i in range(len(res_round)):
				if ex_res[i] != res_round[i]:
					errors += 1

			acc = 1 - errors / len(res_round)
			self.ex_accuracies.append(acc)

		print("Final accuracy: " + str(np.mean(self.ex_accuracies)) + " on " + str(len(self.ex_states)) + " samples")

	def choose_action(self):

		self.saved_state = self.env.get_state()
		self.saved_action = np.random.randint(0, self.env.NB_MOVES)

		return self.saved_action

		#if np.random.random() > self.epsilon:
			#self.saved_action = np.argmax(self.get_qs(self.saved_state))
		#else:
			#self.saved_action = np.random.randint(0, self.env.NB_MOVES)

		#return self.saved_action