import pygame, sys
from pygame.locals import *
import time
import numpy as np

from policy import Policy
from env import Env

from tqdm import tqdm

SHOW_EVERY = -1
SHOW_INFOS = 50

pygame.init()

WIND_ZOOM = 15
NB_STEP_MAX = 400

steps_remaining = NB_STEP_MAX
env = Env()
policy = Policy(env)

windowSurface = pygame.display.set_mode((env.NB_CASES_W*WIND_ZOOM, env.NB_CASES_H*WIND_ZOOM), 0, 32)
pygame.display.set_caption('Curiosity Driven Square')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
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

	if steps_remaining == 0:
		return True
	return env.is_over()

def draw_agent(surface, x, y):
	if x == env.agent.x and y == env.agent.y:
		pygame.draw.rect(surface, BLUE, (x*WIND_ZOOM, y*WIND_ZOOM, WIND_ZOOM-1, WIND_ZOOM-1))
		return True
	return False

def draw_case(surface, x, y):
	color = env.OBJ_2_COLOR[env.onground[x][y]]
	if color == env.OBJ_2_COLOR[env.OBJ_EMPTY]:
		color = env.GROUND_2_COLOR[env.ground[x][y]]

	pygame.draw.rect(surface, color, (x*WIND_ZOOM, y*WIND_ZOOM, WIND_ZOOM-1, WIND_ZOOM-1))

def draw_game(surface):
	for x in range(env.NB_CASES_W):
		for y in range(env.NB_CASES_H):
			if(not draw_agent(surface, x, y)):
				draw_case(surface, x, y)

def draw_state(surface, state, start_x, start_y):
	x = 0
	y = 0
	ground = 0
	for i in range(len(state)):
		if i%2 == 0:
			ground = state[i]
		elif i%2 == 1:
			color = env.OBJ_2_COLOR[state[i]]
			if color == env.OBJ_2_COLOR[env.OBJ_EMPTY]:
				color = env.GROUND_2_COLOR[ground]

			pygame.draw.rect(surface, color, 
				(start_x+x*WIND_ZOOM*3, start_y+y*WIND_ZOOM*3, WIND_ZOOM*3-1, WIND_ZOOM*3-1))

			if i == (len(state)-1)//2+1:
				pygame.draw.rect(surface, BLUE, 
					(start_x+x*WIND_ZOOM*3+4, start_y+y*WIND_ZOOM*3+4, WIND_ZOOM*3-10, WIND_ZOOM*3-10))

			y += 1
			if y == 2*env.VISION_SIZE+1:
				x += 1
				y = 0

ex_state = None
ex_action = None
ex_next = None
ex_res_state = None
ex_acc = None

def calcul_ex(ex_id):
	global ex_state, ex_next, ex_res_state, ex_acc, ex_action

	ex_state = policy.ex_states[ex_id]

	if policy.ex_actions[ex_id] == 0:
		ex_action = "Up"
	elif policy.ex_actions[ex_id] == 1:
		ex_action = "Down"
	elif policy.ex_actions[ex_id] == 2:
		ex_action = "Left"
	elif policy.ex_actions[ex_id] == 3:
		ex_action = "Right"

	ex_next = policy.ex_next_states[ex_id]

	x = policy.state_2_state_res(policy.ex_states[ex_id])
	x = np.concatenate((x, policy.action_2_res(policy.ex_actions[ex_id])))
	res = policy.model.predict(np.array([x]))

	ex_res_state = policy.state_res_2_state(res[0])
	ex_acc = policy.ex_accuracies[ex_id]

def draw_test(surface):
	global ex_state, ex_next, ex_res_state, ex_acc, ex_action

	current_ex = 0
	calcul_ex(current_ex)

	right_down = False
	left_down = False

	textRect.x = 50

	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			if event.type == pygame.KEYDOWN:
				if event.key == K_RIGHT:
					if not right_down:
						right_down = True
						if(current_ex < len(policy.ex_states)-1):
							current_ex += 1
							calcul_ex(current_ex)
				elif event.key == K_LEFT:
					if not left_down:
						left_down = True
						if(current_ex > 0):
							current_ex -= 1
							calcul_ex(current_ex)
			if event.type == pygame.KEYUP:
				if event.key == K_RIGHT:
					right_down = False
				elif event.key == K_LEFT:
					left_down = False

		time.sleep(0.1)
		windowSurface.fill(WHITE)

		draw_state(surface, ex_state, 50, 50)
		draw_state(surface, ex_next, 350, 50)
		draw_state(surface, ex_res_state, 350, 350)

		textRect.y = 350
		text_str = "Action: " + str(ex_action)
		text = basicFont.render(text_str, True, BLACK)
		surface.blit(text, textRect)
		textRect.y = 400
		text_str = "Acc: %.2f" % ex_acc
		text = basicFont.render(text_str, True, BLACK)
		surface.blit(text, textRect)

		pygame.display.update()

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

def show_infos(episode):
	print(episode)

for episode in range(1, policy.EPISODES+1):

	if SHOW_EVERY == -1:
		render = False
	elif episode % SHOW_EVERY == 1:
		render = True
	else:
		render = False

	if episode % SHOW_INFOS == 1:
		show_infos(episode)

	init_game()

	over = False

	if render:
		show_render()

	while not over:

		action = policy.choose_action()
		env.apply_action(action)
		over = is_over()

		policy.update_replay_memory(over)
		policy.train(over)

		steps_remaining -= 1

		if render:
			show_render()

	if policy.END_EPSILON_DECAYING >= episode >= policy.START_EPSILON_DECAYING:
		policy.epsilon -= policy.epsilon_decay_value

policy.test_model()
draw_test(windowSurface)