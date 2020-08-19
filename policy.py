from random import *
import numpy as np

from env import Env

class Policy:

	LEARNING_RATE = 0.4
	DISCOUNT = 0.8
	EPISODES = 10000

	MIN_REWARD_INIT = -0.1
	MAX_REWARD_INIT = 0.1

	START_EPSILON_DECAYING = 1
	END_EPSILON_DECAYING = EPISODES // 2

	ARGMAX_EQ_TH = 0.2

	VOID_ZONE = 0
	NB_ZONES = 5

	def __init__(self, env):

		self.epsilon = 0.5
		self.epsilon_decay_value = self.epsilon/(self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

		self.q_table = []
		self.env = env

		self.saved_action = 0
		self.saved_direct_a = 0
		self.saved_zone_a = 0

		delta = self.MAX_REWARD_INIT - self.MIN_REWARD_INIT

		for i in range(self.env.NB_CASE_TYPES ** self.env.NB_MOVES):
			qt_direct_a = []
			for j in range(self.NB_ZONES ** self.env.NB_MOVES):
				qt_zone_a = []
				for k in range(self.env.NB_MOVES):
					qt_zone_a.append(random()*delta + self.MIN_REWARD_INIT)
				qt_direct_a.append(qt_zone_a)
			self.q_table.append(qt_direct_a)

	def get_direct_a(self):

		state = 0
		for i in range(self.env.NB_MOVES):
			x = self.env.agent.x + self.env.move_shift_x[i]
			y = self.env.agent.y + self.env.move_shift_y[i]

			if(x < 0 or x >= self.env.NB_CASES_W or y < 0 or y >= self.env.NB_CASES_H):
				case_value = self.env.CASE_VOID
			else:
				case_value = self.env.cases[x][y]
			state += (self.env.NB_MOVES ** i) * case_value

		return state


	def interpret_zone_type(self, total, life, danger):

		if(total == 0):
			return self.VOID_ZONE

		zone_id = 1
		if(life > 0):
			zone_id += 1
		if(danger > 0):
			zone_id += 2

		return zone_id


	def calcul_zone_type(self, min_x, max_x, min_y, max_y):

		total = 0
		life = 0
		danger = 0

		for i in range(min_x, max_x+1):
			for j in range(min_y, max_y+1):
				if(0 <= i < self.env.NB_CASES_W and 0 <= j < self.env.NB_CASES_H):
					total += 1
					if(self.env.cases[i][j] == self.env.CASE_LIFE):
						life += 1
					elif(self.env.cases[i][j] == self.env.CASE_DANGER):
						danger += 1

		return self.interpret_zone_type(total, life, danger)

	def get_zone_type(self, move):
		x = self.env.agent.x
		y = self.env.agent.y

		if(move == self.env.MOVE_UP):
			return self.calcul_zone_type(x-1, x+1, y-2, y-1)
		if(move == self.env.MOVE_DOWN):
			return self.calcul_zone_type(x-1, x+1, y+1, y+2)
		if(move == self.env.MOVE_LEFT):
			return self.calcul_zone_type(x-2, x-1, y-1, y+1)
		if(move == self.env.MOVE_RIGHT):
			return self.calcul_zone_type(x+1, x+2, y-1, y+1)

	def get_zone_a(self):
		state = 0
		for i in range(self.env.NB_MOVES):
			zone_type = self.get_zone_type(i)
			state += (self.env.NB_MOVES ** i) * zone_type
		return state

	def choose_argmax(self, moves_q):
		true_max = np.max(moves_q)

		maxs = []
		for i in range(self.env.NB_MOVES):
			if(true_max - moves_q[i] <= self.ARGMAX_EQ_TH):
				maxs.append(i)

		return np.random.choice(maxs)

	def smart_action(self):

		self.saved_direct_a = self.get_direct_a()
		self.saved_zone_a = self.get_zone_a()

		if np.random.random() > self.epsilon:
			self.saved_action = self.choose_argmax(self.q_table[self.saved_direct_a][self.saved_zone_a])
		else:
			self.saved_action = np.random.randint(0, self.env.NB_MOVES)

		return self.saved_action

	def update_q_table(self, reward, over):

		if not over:
			direct_a = self.get_direct_a()
			zone_a = self.get_zone_a()

			max_future_q = np.max(self.q_table[direct_a][zone_a])
			current_q = self.q_table[self.saved_direct_a][self.saved_zone_a][self.saved_action]
			new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT * max_future_q)

			self.q_table[self.saved_direct_a][self.saved_zone_a][self.saved_action] = new_q

	def choose_action(self):
		return self.smart_action()