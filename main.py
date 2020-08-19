import pygame, sys
from pygame.locals import *
import time

from policy import Policy
from env import Env

SHOW_EVERY = 500

pygame.init()

WIND_ZOOM = 15
NB_STEP_MAX = 80

steps_remaining = NB_STEP_MAX
env = Env()
policy = Policy(env)

windowSurface = pygame.display.set_mode((env.NB_CASES_W*WIND_ZOOM, env.NB_CASES_H*WIND_ZOOM), 0, 32)
pygame.display.set_caption('Curiosity Driven Square')

BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

basicFont = pygame.font.SysFont(None, 24)
text = basicFont.render('Hello world!', True, BLACK)
textRect = text.get_rect()
textRect.x = 10
textRect.y = 10

reward = 0

def init_game():
	global steps_remaining, env, policy

	steps_remaining = NB_STEP_MAX
	env = Env()
	policy.env = env

def update_game():
	global reward

	a = policy.choose_action()
	reward = env.apply_action(a)

def is_over():

	if(steps_remaining == 0):
		return True
	return env.is_over()

def draw_agent(surface, x, y):
	if(x == env.agent.x and y == env.agent.y):
		pygame.draw.rect(surface, BLUE, (x*WIND_ZOOM, y*WIND_ZOOM, WIND_ZOOM-1, WIND_ZOOM-1))
		return True
	return False

def draw_case(surface, x, y):
	color = BLACK

	if(env.cases[x][y] == env.CASE_EMPTY):
		color = GRAY
	elif(env.cases[x][y] == env.CASE_LIFE):
		color = GREEN
	elif(env.cases[x][y] == env.CASE_DANGER):
		color = RED

	pygame.draw.rect(surface, color, (x*WIND_ZOOM, y*WIND_ZOOM, WIND_ZOOM-1, WIND_ZOOM-1))

def draw_game(surface):
	for x in range(env.NB_CASES_W):
		for y in range(env.NB_CASES_H):
			if(not draw_agent(surface, x, y)):
				draw_case(surface, x, y)

def show_render():
	for event in pygame.event.get():
		if event.type == QUIT:
			pygame.quit()
			print("247")
			print(policy.q_table[247][0])
			print("95")
			print(policy.q_table[95][0])
			print("149")
			print(policy.q_table[149][0])
			print("182")
			print(policy.q_table[182][0])
			sys.exit()

	time.sleep(0.1)
	windowSurface.fill(WHITE)

	draw_game(windowSurface)

	text_str = str(env.agent.life)
	text = basicFont.render(text_str, True, BLACK)
	windowSurface.blit(text, textRect)

	pygame.display.update()

for episode in range(policy.EPISODES):
	
	if episode % SHOW_EVERY == SHOW_EVERY-1:
		render = True
		print(episode)
	else:
		render = False

	init_game()

	over = False

	if(render and policy.END_EPSILON_DECAYING < episode):
		show_render()

	while not over:

		update_game()

		steps_remaining -= 1

		over = is_over()

		policy.update_q_table(reward, over)
		
		if(render and policy.END_EPSILON_DECAYING < episode):
			show_render()

	if policy.END_EPSILON_DECAYING >= episode >= policy.START_EPSILON_DECAYING:
		policy.epsilon -= policy.epsilon_decay_value

		