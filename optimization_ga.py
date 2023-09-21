import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import pandas as pd
from math import fabs,sqrt
import glob, os

experiment = 'GA_optimization' # name of the experiment
headless = True # True for not using visuals, false otherwise
enemies = [1,2,7]
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test
npop = 50 # size of population
gens = 100 # max number of generations
mutation = 0.1 # mutation probability 

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment)
    
ENV = Environment(experiment_name=experiment,
                enemies=[enemies],
                playermode=playermode,
                player_controller=player_controller(n_hidden_nodes),
                enemymode=enemymode,
                level=2,
                speed="fastest",
                visuals=False)

ENV.state_to_log() # checks environment state
n_gen = (ENV.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector

def initialize_population(n):
    return np.random.uniform(lb_w, ub_w, (n, n_gen))

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# limits
def check_bounds(x):
    if x>ub_w:
        return ub_w
    elif x<lb_w:
        return lb_w
    else:
        return x
    
# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

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

# crossover
def crossover(p1, p2):

    n_offspring = np.random.randint(1,3+1, 1)[0]
    offspring = np.zeros( (n_offspring, n_gen) )

    for f in range(0, n_offspring):

        cross_prop = np.random.uniform(0,1)
        offspring[f] = p1*cross_prop+p2*(1-cross_prop)

        # mutation
        for i in range(0,len(offspring[f])):
            if np.random.uniform(0 ,1) <= mutation:
                offspring[f][i] = offspring[f][i]+np.random.normal(0, 1)

        offspring[f] = np.array(list(map(lambda y: check_bounds(y), offspring[f])))

    return offspring
    
def create_offspring(pop, fitness, probabilities, fps_var, fps_transpose, ranking_var, sampling):
    couples = select_parents(pop, fitness, fps_var, fps_transpose, ranking_var, sampling, probabilities) #default, transpose, windowing, scaling
    total_offspring = np.zeros((0, n_gen))
    for parents in couples:
        offspring = crossover(parents[0], parents[1])
        total_offspring = np.vstack((total_offspring, offspring))
        
    return total_offspring

for enemy in enemies:
    ENV.update_parameter('enemies',[enemy])
    # initializes simulation in individual evolution mode, for single static enemy.
    ini = time.time()  # sets time marker
    last_best = 0
        
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])
    pop = initialize_population(npop)
    fitness_gen = evaluate(pop)
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen)])
    best_txt = ''
    for gen in range(1, gens+1):
        print('Current generation:', gen)
        offspring = create_offspring(pop, fitness_gen, probabilities="ranking", fps_var="scaling", fps_transpose = 10, ranking_var="exponential", sampling="sus")
        fitness_offspring = evaluate(offspring)
        pop = np.vstack((pop, offspring))
        fitness_gen = np.append(fitness_gen, fitness_offspring)
        
        best_idx = np.argmax(fitness_gen) #best solution in generation
        fitness_gen[best_idx] = float(evaluate(np.array([pop[best_idx] ]))[0]) # repeats best eval, for stability issues
        best_sol = fitness_gen[best_idx]
        best_txt = pop[best_idx]
        
        # selection
        fitness_gen_cp = fitness_gen
        fitness_gen_norm =  np.array(list(map(lambda y: norm(y,fitness_gen_cp), fitness_gen))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fitness_gen_norm)/(fitness_gen_norm).sum()
        chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
        chosen = np.append(chosen[1:], best_idx)
        pop = pop[chosen]
        fitness_gen = fitness_gen[chosen]
        
        std = np.std(fitness_gen)
        mean = np.mean(fitness_gen)
        print('Scores:\nBest: {:.2f}, Mean: {:.2f}, Std: {:.2f}'.format(best_sol, mean, std))
        
        results.loc[len(results)] = np.array([gen, best_sol, mean, std])
        
    np.savetxt(experiment+'/best_'+ str(enemy) +'.txt',best_txt)
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    
    