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

def selection_fps(fit_vals, fps_var="default", fps_transpose=10):
    # Fitness Proportional Selection (FPS) has 4 variations: default, transpose, windowing, scaling
    if fps_var == 'transpose':
        fit_vals += fps_transpose
    elif fps_var == 'windowing':
        fit_vals -= np.min(fit_vals)
    elif fps_var == 'scaling':
        mean, std = np.mean(fit_vals), np.std(fit_vals)
        fit_vals = np.array([max(f - (mean - 2*std),0) for f in fit_vals])
        
    selection_probabilities = np.array(fit_vals) / sum(fit_vals)
    return selection_probabilities
        
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
        
def selection_ranking(pop, fit_vals, ranking_var='linear'):
    sorted_indices = np.argsort(fit_vals)[::-1]
    selection_probabilities = [selection_probability(len(pop), 1.5, i, method=ranking_var) for i in range(len(pop))]
    reordered_probabilities = [selection_probabilities[i] for i in sorted_indices]

    return np.array(reordered_probabilities)


def select_parents(pop, fit_vals, fps_var="default", fps_transpose=10, ranking_var="exponential", sampling="sus", probabilities="fps"):
    """
    Parameters:
    - pop: The population
    - fit_vals: The fitness values of the current population
    - fps_var: The chosen variation of the FPS methods (default, transpose, windowing, scaling)
    - fps_transpose: The transpose value for the FPS transpose method (defualt = 10)
    - ranking_var: The chosen variatio of the ranking methods (linear, exponential)
    - sampling: The chosen sampling method (roulette, sus)
    - probabilities: The chosen method for calculating the selection probabilities (fps, ranking, )
    """
    num_couples = round(len(pop)/2)
    couples = np.zeros((num_couples, 2, n_gen))
    selection_probabilities = np.zeros((len(pop), 2))
    
    min_fitness = np.min(fit_vals)
    adjusted_fitness = fit_vals - min_fitness + 0.0000001
    
    if probabilities == 'fps':
        selection_probabilities = selection_fps(adjusted_fitness, fps_var, fps_transpose)
    elif probabilities == 'ranking':
        selection_probabilities = selection_ranking(pop, adjusted_fitness, ranking_var='linear')
            
    if sampling == 'sus':
        num_parents = num_couples * 2  # We select twice the number of parents since each couple has 2 parents
        parents = np.zeros((num_parents, n_gen))
        current_member = i = 0
        r = np.random.uniform(0, 1 / num_parents)
        cumulative_probabilities = np.cumsum(selection_probabilities)
        
        while current_member < num_parents:
            while r <= cumulative_probabilities[i]:
                parents[current_member] = pop[i]
                r += 1 / num_parents
                current_member += 1
            i += 1

        couples = parents.reshape(num_couples, 2, 265)

        return couples
    else:
        for i in range(num_couples):
            idx = np.random.choice(len(pop), 2, p=selection_probabilities)
            couples[i] = [pop[idx[0]], pop[idx[1]]]
        
    return couples
       
def create_offspring(pop, fitness, probabilities, fps_var, fps_transpose, ranking_var, sampling):
    couples = select_parents(pop, fitness, fps_var, fps_transpose, ranking_var, sampling, probabilities) #default, transpose, windowing, scaling
    for parents in couples:
        return
    
    
results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])
pop = initialize_population(npop)
fitness_gen = evaluate(pop)
results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen)])
create_offspring(pop, fitness_gen, probabilities="ranking", fps_var="scaling", fps_transpose = 10, ranking_var="exponential", sampling="sus")

# for gen in range(1, gens+1):
#     offspring = create_offspring(pop, fitness_gen)
