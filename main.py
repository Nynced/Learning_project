import pygame, sys
from pygame.locals import *
import time

from policy import Policy
from env import Env

from tqdm import tqdm

SHOW_EVERY = 50

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

def init_game():
	global steps_remaining, env, policy

	steps_remaining = NB_STEP_MAX
	env = Env()
	policy.env = env

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
			sys.exit()

	time.sleep(0.1)
	windowSurface.fill(WHITE)

	draw_game(windowSurface)

	text_str = str(env.agent.life)
	text = basicFont.render(text_str, True, BLACK)
	windowSurface.blit(text, textRect)

	pygame.display.update()

for episode in tqdm(range(1, policy.EPISODES+1), ascii=True, unit='episodes'):

	if episode % SHOW_EVERY == 1:
		render = True
	else:
		render = False

	init_game()

	over = False

	if(render):
		show_render()

	while not over:

		action = policy.choose_action()
		reward = env.apply_action(action)
		over = is_over()

		policy.update_replay_memory(reward, over)
		policy.train(over)

		steps_remaining -= 1

		if(render):
			show_render()

	if policy.END_EPSILON_DECAYING >= episode >= policy.START_EPSILON_DECAYING:
		policy.epsilon -= policy.epsilon_decay_value