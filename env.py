from random import *
import numpy as np

from agent import Agent

class Env:

	NB_CASES_W = 41
	NB_CASES_H = 41

	VISION_SIZE = 2
	TOTAL_VISION = (2*VISION_SIZE+1)*(2*VISION_SIZE+1)
	STATE_SIZE = TOTAL_VISION * 2

	MOVE_UP = 0
	MOVE_DOWN = 1
	MOVE_LEFT = 2
	MOVE_RIGHT = 3
	NB_MOVES = 4

	move_shift_x = [0, 0, -1, 1]
	move_shift_y = [-1, 1, 0, 0]

	GROUND_VOID = 0
	GROUND_GRASS = 1
	GROUND_DIRT = 2
	NB_GROUND_TYPES = 3

	OBJ_EMPTY = 0
	OBJ_LIFE = 1
	OBJ_DANGER = 2
	OBJ_CLONE = 3
	NB_OBJ_TYPES = 4

	GROUND_2_COLOR = [(0, 0, 0),
		(13, 64, 13),
	    (50, 38, 13)]

	OBJ_2_COLOR = [(0, 0, 0),
		(0, 255, 0),
        (255, 0, 0),
        (0, 255, 255)]

	def __init__(self):

		self.ground = []
		self.onground = []
		self.agent = Agent(self.NB_CASES_W//2, self.NB_CASES_H//2)

		for i in range(self.NB_CASES_W):
			ground_line = []
			onground_line = []
			for j in range(self.NB_CASES_H):
				ground_line.append(self.init_ground())
				onground_line.append(self.init_onground())
			self.ground.append(ground_line)
			self.onground.append(onground_line)

		self.onground[self.agent.x][self.agent.y] = self.OBJ_EMPTY

	def init_ground(self):
		return np.random.randint(0, self.NB_GROUND_TYPES-1)+1

	def init_onground(self):

		r = int(random() * 100)

		if 0 <= r < 20 :
			return self.OBJ_LIFE
		elif 20 <= r < 40:
			return self.OBJ_DANGER
		elif r == 40:
			return self.OBJ_CLONE
		else:
			return self.OBJ_EMPTY

	def is_over(self):

		return self.agent.life == 0

	def get_state(self):
		state = []

		i = 0
		for x in range(self.agent.x-self.VISION_SIZE, self.agent.x+self.VISION_SIZE+1):
			j = 0
			for y in range(self.agent.y-self.VISION_SIZE, self.agent.y+self.VISION_SIZE+1):
				if(x < 0 or x >= self.NB_CASES_W
				or y < 0 or y >= self.NB_CASES_H):
					state.append(self.GROUND_VOID)
					state.append(self.OBJ_EMPTY)
				else:
					state.append(self.ground[x][y])
					state.append(self.onground[x][y])
				j += 1
			i += 1

		return state

	def update_cases(self):

		x = self.agent.x
		y = self.agent.y
		case = self.onground[x][y]
		
		if case == self.OBJ_LIFE:
			self.agent.life += 1
			self.onground[x][y] = self.OBJ_EMPTY
		elif case == self.OBJ_DANGER:
			self.agent.life -= 1
		elif case == self.OBJ_CLONE:
			self.onground[x][y] = self.OBJ_EMPTY
			for i in (x-1, x+1):
				for j in (y-1, y+1):
					if(0 <= i < self.NB_CASES_W
					and 0 <= j < self.NB_CASES_H):
						self.onground[i][j] = self.OBJ_CLONE


	def apply_action(self, a):

		if a == self.MOVE_UP:
			if self.agent.y > 0:
				self.agent.y -= 1
		elif a == self.MOVE_DOWN:
			if self.agent.y < self.NB_CASES_H-1:
				self.agent.y += 1
		elif a == self.MOVE_LEFT:
			if self.agent.x > 0:
				self.agent.x -= 1
		elif a == self.MOVE_RIGHT:
			if self.agent.x < self.NB_CASES_W-1:
				self.agent.x += 1

		self.update_cases()