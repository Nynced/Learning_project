import random
import numpy as np

from env import Env

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

from collections import deque

import time

class Policy:

	REPLAY_MEMORY_SIZE = 50000
	MIN_REPLAY_MEMORY_SIZE = 1000
	MINIBATCH_SIZE = 64
	UPDATE_TARGET_EVERY = 5
	MODEL_NAME = "256_64"

	LEARNING_RATE = 0.4
	DISCOUNT = 0.85
	EPISODES = 500

	MIN_REWARD_INIT = -0.1
	MAX_REWARD_INIT = 0.1

	START_EPSILON_DECAYING = 1
	END_EPSILON_DECAYING = EPISODES // 2

	ARGMAX_EQ_TH = 0.2

	VOID_ZONE = 0
	NB_ZONES = 5

	def __init__(self, env):

		self.env = env

		self.saved_action = 0
		self.saved_state = None

		self.epsilon = 1.0
		self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

		self.model = self.create_model()

		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

		self.target_update_counter = 0

	def create_model(self):

		model = Sequential()

		model.add(Conv2D(256, (3,3), input_shape=(2*self.env.VISION_SIZE+1, 2*self.env.VISION_SIZE+1, 3)))
		model.add(Activation("relu"))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(64))

		model.add(Dense(self.env.NB_MOVES, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

		return model

	def update_replay_memory(self, reward, over):
		transition = (self.saved_state, self.saved_action, reward, self.env.get_state(), over)

		self.replay_memory.append(transition)

	def get_qs(self, state):
		return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

	def train(self, terminal_state):
		if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

		current_states = np.array([transition[0] for transition in minibatch])/255
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])/255
		future_qs_list = self.target_model.predict(new_current_states)

		X = []
		y = []

		for index, (current_state, action, reward, new_current_state, over) in enumerate(minibatch):

			if not over:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.DISCOUNT * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			X.append(current_state)
			y.append(current_qs)

		self.model.fit(np.array(X)/255, np.array(y), batch_size = self.MINIBATCH_SIZE,
			verbose = 0, shuffle=False if terminal_state else None)

		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > self.UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

	def update_q_table(self, reward, over):

		if not over:
			direct_a = self.get_direct_a()
			zone_a = self.get_zone_a()

			max_future_q = np.max(self.q_table[direct_a][zone_a])
			current_q = self.q_table[self.saved_direct_a][self.saved_zone_a][self.saved_action]
			new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)

			self.q_table[self.saved_direct_a][self.saved_zone_a][self.saved_action] = new_q

	def choose_action(self):
		self.saved_state = self.env.get_state()

		if np.random.random() > self.epsilon:
			self.saved_action = np.argmax(self.get_qs(self.saved_state))
		else:
			self.saved_action = np.random.randint(0, self.env.NB_MOVES)

		return self.saved_action