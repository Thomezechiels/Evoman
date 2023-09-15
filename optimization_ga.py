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

def select_parents(pop, fit_vals):
    num_couples = round(len(pop)/2) 
    selection_probabilities = np.array(fit_vals) / sum(fit_vals)
    couples = []

    for _ in range(num_couples):
        parent1_idx = np.random.choice(len(pop), p=selection_probabilities)
        parent2_idx = np.random.choice(len(pop), p=selection_probabilities)
        couples.append((pop[parent1_idx], pop[parent2_idx]))

    return couples

def create_offspring(pop, fitness):
    couples = select_parents(pop, fitness)
    for parents in couples:
        
    
    
results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])
pop = initialize_population(npop)
fitness_gen = evaluate(pop)
results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen)])
create_offspring(pop, fitness_gen)
# for gen in range(1, gens+1):
#     offspring = create_offspring(pop, fitness_gen)
