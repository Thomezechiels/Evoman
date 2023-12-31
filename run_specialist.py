#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'GA_optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

# tests saved demo solutions for each enemy
enemies = [1,2,7]
for en in enemies:

	#Update the enemy
	env.update_parameter('enemies',[en])

	# Load specialist controller
	sol = np.loadtxt(experiment_name + '/best_DGA_' + str(en) + '.txt')
	# env.play(sol)
	sol = [sol]
	for i in range(1):
		solution = sol[i]
		print('\n LOADING SAVED SPECIALIST SOLUTION {} FOR ENEMY '.format(i) +str(en)+' \n')
		env.play(solution)
