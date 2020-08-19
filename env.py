from random import *

from agent import Agent

class Env:

	NB_CASES_W = 41
	NB_CASES_H = 41

	MOVE_UP = 0
	MOVE_DOWN = 1
	MOVE_LEFT = 2
	MOVE_RIGHT = 3
	NB_MOVES = 4

	move_shift_x = [0, 0, -1, 1]
	move_shift_y = [-1, 1, 0, 0]

	CASE_VOID = 0
	CASE_EMPTY = 1
	CASE_LIFE = 2
	CASE_DANGER = 3
	NB_CASE_TYPES = 4

	def __init__(self):

		self.cases = []
		self.agent = Agent(self.NB_CASES_W//2, self.NB_CASES_H//2)

		for i in range(self.NB_CASES_W):
			cases_line = []
			for j in range(self.NB_CASES_H):
				cases_line.append(self.init_case())
			self.cases.append(cases_line)

		self.cases[self.agent.x][self.agent.y] = self.CASE_EMPTY

	def init_case(self):

		r = int(random()* (self.NB_CASE_TYPES-1) * 2)

		if(r == self.CASE_LIFE or r == self.CASE_DANGER):
			return r
		else:
			return self.CASE_EMPTY

	def is_over(self):

		return self.agent.life == 0

	def update_cases(self):

		case = self.cases[self.agent.x][self.agent.y]
		
		if(case == self.CASE_LIFE):
			self.agent.life += 1
			self.cases[self.agent.x][self.agent.y] = self.CASE_EMPTY
			return 1
		elif(case == self.CASE_DANGER):
			self.agent.life -= 1
			return -1

		return 0

	def apply_action(self, a):

		if(a == self.MOVE_UP):
			if(self.agent.y > 0):
				self.agent.y -= 1
		elif(a == self.MOVE_DOWN):
			if(self.agent.y < self.NB_CASES_H-1):
				self.agent.y += 1
		elif(a == self.MOVE_LEFT):
			if(self.agent.x > 0):
				self.agent.x -= 1
		elif(a == self.MOVE_RIGHT):
			if(self.agent.x < self.NB_CASES_W-1):
				self.agent.x += 1

		return self.update_cases()