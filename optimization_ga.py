import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import pandas as pd
from math import fabs,sqrt
import glob, os

experiment = 'optimization_test' # name of the experiment
headless = True # True for not using visuals, false otherwise
enemies = [8]
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test
npop = 100 # size of population
gens = 30 # max number of generations
mutation = 0.2 # mutation probability 

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment)


# initializes simulation in individual evolution mode, for single static enemy.
ENV = Environment(experiment_name=experiment,
                  enemies=enemies,
                  playermode=playermode,
                  player_controller=player_controller(n_hidden_nodes),
                  enemymode=enemymode,
                  level=2,
                  speed="fastest",
                  visuals=False)

ENV.state_to_log() # checks environment state
ini = time.time()  # sets time marker
n_gen = (ENV.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector
last_best = 0

def initialize_population(n):
    return np.random.uniform(lb_w, ub_w, (n, n_gen))

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def evaluate(x):
    return np.array(list(map(lambda y: simulation(ENV, y), x)))

def selection_fps(pop, fit_vals, num_couples, fps_var="default", fps_transpose=10):
    # Fitness Proportional Selection (FPS) has 4 variations: default, transpose, windowing, scaling
    couples = np.zeros(num_couples)
    if fps_var == 'transpose':
        fit_vals += fps_transpose
    elif fps_var == 'windowing':
        fit_vals -= np.min(fit_vals)
    elif fps_var == 'scaling':
        mean, std = np.mean(fit_vals), np.std(fit_vals)
        fit_vals = np.array([max(f - (mean - 2*std),0) for f in fit_vals])
        
    selection_probabilities = np.array(fit_vals) / sum(fit_vals)
    for i in range(num_couples):
        idx = np.random.choice(len(pop), 2, p=selection_probabilities)
        couples[i] = (pop[idx[0]], pop[idx[1]])
        
import math

def selection_probability(mu, s, rank, method='linear'):
    if rank < 0 or rank >= mu:
        raise ValueError("Rank must be between 0 and mu - 1")

    if method == 'linear':
        if not (1 < s <= 2):
            raise ValueError("s must be between 1 < s <= 2")

        return (2 - s) / mu + 2 * rank * (s - 1) / (mu * (mu - 1))

    elif method == 'exponential':
        c = mu / (1 - math.exp(-mu))  # Calculate the exponential ranking parameter 'c'
        return (1 - math.exp(-rank)) / c
        
def selection_ranking(pop, fit_vals, num_couples, method='linear'):
    # Ranking Selection (RS) has 2 variations: linear, exponential
    couples = np.zeros((num_couples, 2))
    selection_probabilities = [selection_probability(len(pop), 1.5, i, method=method) for i in range(len(pop))]
    
    for i in range(num_couples):
        idx = np.random.choice(len(pop), 2, p=selection_probabilities)
        couples[i] = (pop[idx[0]], pop[idx[1]])

    return couples


def select_parents(pop, fit_vals, fps_var, fps_transpose, method="fps"):
    num_couples = round(len(pop)/2)
    
    if method == 'fps':
        return selection_fps(pop, fit_vals, num_couples, fps_var, fps_transpose)
    elif method == 'ranking':
        return selection_ranking(pop, fit_vals)
        
        
def create_offspring(pop, fitness):
    couples = select_parents(pop, fitness)
    for parents in couples:
        return
    
    
results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])
pop = initialize_population(npop)
fitness_gen = evaluate(pop)
results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen)])
create_offspring(pop, fitness_gen)
# for gen in range(1, gens+1):
#     offspring = create_offspring(pop, fitness_gen)
