import sys
from evoman.environment import Environment
from controller import player_controller

# imports other libs
import time
import numpy as np
import pandas as pd
from math import fabs,sqrt
from scipy.stats import wasserstein_distance
import glob, os
import random
import matplotlib.pyplot as plt

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

def fitness_single(self):
        return 0.7*(100 - self.get_enemylife()) + 0.3*self.get_playerlife() - 0.5 * np.log(self.get_time())
        # return self.get_playerlife() - self.get_enemylife()

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

def selection_fps(fit_vals, fps_var="default", fps_transpose=10):
    # Fitness Proportional Selection (FPS) has 4 variations: default, transpose, windowing, scaling
    if fps_var == 'transpose':
        fit_vals += fps_transpose
    elif fps_var == 'windowing':
        fit_vals -= np.min(fit_vals)
    elif fps_var == 'scaling':
        mean, std = np.mean(fit_vals), np.std(fit_vals)
        fit_vals = np.vectorize(lambda f: max(f - (mean - 2*std), 0))
        
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

    n_offspring = np.random.randint(1,4)
    offspring = np.zeros((n_offspring, n_gen))

    for f in range(0, n_offspring):
        cross_prop = np.random.uniform(0,1)
        offspring[f] = p1*cross_prop+p2*(1-cross_prop)
  
        # mutation
        for i in range(0,len(offspring[f])):
            if np.random.uniform(0 ,1) <= mutation:
                offspring[f][i] = offspring[f][i]+np.random.normal(0, 0.1)

        offspring[f] = np.array(list(map(lambda y: check_bounds(y), offspring[f])))

    return offspring
    
def create_offspring(pop, fitness, probabilities, fps_var, fps_transpose, ranking_var, sampling):
    couples = select_parents(pop, fitness, fps_var, fps_transpose, ranking_var, sampling, probabilities) #default, transpose, windowing, scaling
    total_offspring = np.zeros((0, n_gen))
    for parents in couples:
        offspring = crossover(parents[0], parents[1])
        total_offspring = np.vstack((total_offspring, offspring))
        
    return total_offspring

def exchange_information(subpops, fitness_subpops, migration_rate=0.1):
    n_subpops = len(subpops)
    n_individuals = np.array([len(s) for s in subpops])
    n_to_migrate = (migration_rate * n_individuals).astype(int)
    migrations = [[] for _ in range(n_subpops)]
    migrations_fitness = [[] for _ in range(n_subpops)]
    
    for idx, subpop in enumerate(subpops):
        migrants_idx = np.random.choice(len(subpop), n_to_migrate[idx], replace=False)
        migrants = subpop[migrants_idx]
        migrants_fitness = fitness_subpops[idx][migrants_idx]
        for i, migrant in enumerate(migrants):
            i_adjusted = (i + idx) % n_subpops
            migrations[i_adjusted].append(migrant)
            migrations_fitness[i_adjusted].append(migrants_fitness[i])
        subpops[idx] = np.delete(subpop, migrants_idx, axis=0)
        fitness_subpops[idx] = np.delete(fitness_subpops[idx], migrants_idx, axis=0)
        
    for idx, subpop in enumerate(subpops):
        if len(migrations[idx]) > 0:
            subpops[idx] = np.append(subpop, migrations[idx], axis=0)
            fitness_subpops[idx] = np.append(fitness_subpops[idx], migrations_fitness[idx], axis=0)
        
def print_size_subpops(subpops):
    print('Subpops sizes:')
    for idx, pop in enumerate(subpops):
        print('Subpop {}: {}'.format(idx, len(pop)))
        
def calculate_diversity(pop):
    histograms = [w / w.sum() for w in pop]

    # Calculate diversity using Wasserstein distance
    total_diversity = 0.0

    # Calculate pairwise Wasserstein distances and sum them up
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            emd = wasserstein_distance(histograms[i], histograms[j])
            total_diversity += emd
            
    return total_diversity / (len(pop) * (len(pop) - 1) / 2)

def round_robin(pop, fit_pop, tournament = "random"):
    num_individuals = len(pop)
    robin_scores = [0] * num_individuals
    
    if tournament == "random":
        for i in range(num_individuals):
            for _ in range(10):
                result_i = fit_pop[i]
                result_j = fit_pop[np.random.randint(0,pop.shape[0], 1)]
                
                if result_i > result_j:
                    robin_scores[i] += 1

    else:
        for i in range(num_individuals):
            for j in range(i+1, num_individuals):
                result_i = fit_pop[i]
                result_j = fit_pop[j]
                
                if result_i > result_j:
                    robin_scores[i] += 1
                elif result_i < result_j:
                    robin_scores[j] += 1
                # In case of a tie, no points are awarded
    
    return robin_scores

def select_robin(scores, gen_size):
    total_scores = sum(scores)
    print(scores)
    probabilities = scores / total_scores
    
    selected_indices = []
    for _ in range(gen_size):
        rand_num = random.uniform(0, 1)
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= rand_num:
                selected_indices.append(i)
                break

    return selected_indices

def plot_diversity(results):
    generation = results['gen']
    diversity = results['diversity']

    # Create a line plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(generation, diversity, marker='o', linestyle='-')
    plt.title('Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.grid(True)

    # Show the plot
    plt.show()

def train_specialist_GA(env, enemies, experiment, test=False):
    for enemy in enemies:
        env.update_parameter('enemies',[enemy])
        ini = time.time()
        results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
        
        pop = initialize_population(npop)
        fitness_gen = evaluate(env, pop)
        results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
        best_txt = ''
        
        for gen in range(1, gens+1):
            print('Current generation:', gen)
            offspring = create_offspring(pop, fitness_gen, probabilities="ranking", fps_var="scaling", fps_transpose = 10, ranking_var="linear", sampling="sus")
            fitness_offspring = evaluate(env, offspring)
            pop = np.vstack((pop, offspring))
            fitness_gen = np.append(fitness_gen, fitness_offspring)
            
            best_idx = np.argmax(fitness_gen) #best solution in generation
            fitness_gen[best_idx] = float(evaluate(env, np.array([pop[best_idx] ]))[0]) # repeats best eval, for stability issues
            best_sol = fitness_gen[best_idx]
            best_txt = pop[best_idx]
            
            #survival selection 
            scores = np.array(round_robin(pop, fitness_gen, tournament = "random"))    
            selected = np.array(select_robin(scores, gen_size = npop))
            selected = np.append(selected[1:], best_idx)
            pop = pop[selected]
            fitness_gen = fitness_gen[selected]
           
            std = np.std(fitness_gen)
            mean = np.mean(fitness_gen)
            div = calculate_diversity(pop)
            print('Scores:\nBest: {:.2f}, Mean: {:.2f}, Std: {:.2f}\nDiversity: {:.4f}'.format(best_sol, mean, std, div))
            
            
            results.loc[len(results)] = np.array([gen, best_sol, mean, std, div])
            
        print('Final eval best solution:', simulation(env, best_txt))    
        calculate_diversity(pop)
        plot_diversity(results)
        if not test:
            np.savetxt(experiment+'/best_'+ str(enemy) +'.txt',best_txt)
        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    
def train_specialist_DGA(env, enemies, experiment, n_subpops, test=False):
    for enemy in enemies:
        env.update_parameter('enemies',[enemy])
        ini = time.time() 
        results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
        
        pop = initialize_population(npop)
        subpops = np.array_split(pop, n_subpops, axis=0)
        fitness_subpops = [evaluate(env, x) for x in subpops]
        fitness_gen = np.concatenate(fitness_subpops)
        results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
        best_txt = ''
        
        for gen in range(1, gens+1):
            print('Current generation:', gen)
            
            for idx, subpop in enumerate(subpops):
                sub_pop_size = len(subpop)
                fitness_subpop = fitness_subpops[idx]
                exchange_information(subpops, fitness_subpops, migration_rate=migration)
                offspring = create_offspring(subpop, fitness_subpop, probabilities="ranking", fps_var="scaling", fps_transpose = 10, ranking_var="exponential", sampling="sus")
                fitness_offspring = evaluate(env, offspring)
                cur_pop = np.vstack((subpop, offspring))
                fitness_subpop = np.append(fitness_subpop, fitness_offspring)
                best_idx = np.argmax(fitness_subpop) #best solution in generation
                fitness_subpop[best_idx] = float(evaluate(env, np.array([cur_pop[best_idx] ]))[0]) # repeats best eval

                
                scores = np.array(round_robin(cur_pop, fitness_subpop, tournament = "random"))    
                selected = np.array(select_robin(scores, gen_size = sub_pop_size))
                selected = np.append(selected[1:], best_idx)
                # print(selected)
                cur_pop = cur_pop[selected]
                fitness_subpop = fitness_subpop[selected]
                
                # fitness_gen_cp = fitness_subpop
                # fitness_gen_norm =  np.array(list(map(lambda y: norm(y,fitness_gen_cp), fitness_subpop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
                # probs = (fitness_gen_norm)/(fitness_gen_norm).sum()
                # chosen = np.random.choice(cur_pop.shape[0], sub_pop_size , p=probs, replace=False)
                # chosen = np.append(chosen[1:], best_idx)
                # cur_pop = cur_pop[chosen]
                # fitness_subpop = fitness_subpop[chosen]
                fitness_subpops[idx] = fitness_subpop
                subpops[idx] = cur_pop
                
                std = np.std(fitness_subpop)
                mean = np.mean(fitness_subpop)
                best = np.max(fitness_subpop)
                # print('Generation: {}\nSubpop: {}\nBest: {:.2f}, Mean: {:.2f}, Std: {:.2f}'.format(gen, idx, best, mean, std))
                
            fitness_gen = np.concatenate(fitness_subpops)
            pop = np.concatenate(subpops)
            best_txt = pop[np.argmax(fitness_gen)]
            best = np.max(fitness_gen)
            mean = np.mean(fitness_gen)
            std = np.std(fitness_gen)
            div = calculate_diversity(pop)
            print('Generation: {}, Diversity: {}'.format(gen, div))
            results.loc[len(results)] = np.array([gen, best, mean, std, div])
            
        print('Final eval best solution:', simulation(env, best_txt))
        
        div = calculate_diversity(pop)   
        plot_diversity(results)
        print('Average diversity final generation: {}'.format(div)) 
        if not test:
            np.savetxt(experiment+'/best_'+ str(enemy) +'.txt',best_txt)
        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')

experiment = 'GA_optimization' # name of the experiment
headless = True # True for not using visuals, false otherwise
enemies = [2]
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test
npop = 100 # size of population
gens = 50 # max number of generations
mutation = 0.1 # mutation probability
migration = 0.04
n_subpops = 8

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment)
    
ENV = Environment(experiment_name=experiment,
                enemies=enemies,
                playermode=playermode,
                player_controller=player_controller(n_hidden_nodes),
                enemymode=enemymode,
                level=2,
                speed="fastest",
                visuals=False)

import types

ENV.fitness_single = types.MethodType(fitness_single, ENV)


ENV.state_to_log() # checks environment state
n_gen = (ENV.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector    
  
train_specialist_DGA(ENV, enemies, experiment, n_subpops, test=True)
# train_specialist_GA(ENV, enemies, experiment, test=True)              

    
    