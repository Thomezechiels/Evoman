import sys
from evoman.environment import Environment
from controller import player_controller

import optuna
import types
import time
import math
import numpy as np
import pandas as pd
from math import fabs,sqrt
from scipy.stats import wasserstein_distance
import glob, os
import random
import matplotlib.pyplot as plt

def initialize_population(n):
    return np.random.uniform(lb_w, ub_w, (n, n_gen))

def fitness_single(self):
        # return 0.7*(100 - self.get_enemylife()) + 0.3*self.get_playerlife() - 0.5 * np.log(self.get_time())
        return 100 + self.get_playerlife() - self.get_enemylife()

def selection_ranking(pop, fit_vals, ranking_var='linear'):
    sorted_indices = np.argsort(fit_vals)[::-1]
    selection_probabilities = [selection_probability(len(pop), 1.5, i, method=ranking_var) for i in range(len(pop))]
    reordered_probabilities = [selection_probabilities[i] for i in sorted_indices]
    return np.array(reordered_probabilities)

def selection_fps(fit_vals, fps_var="default", fps_transpose=10):
    # Fitness Proportional Selection (FPS) has 4 variations: default, transpose, windowing, scaling
    if fps_var == 'transpose':
        fit_vals += fps_transpose
    elif fps_var == 'windowing':
        fit_vals -= np.min(fit_vals)
    elif fps_var == 'scaling':
        mean, std = np.mean(fit_vals), np.std(fit_vals)
        fit_vals = [max(f - (mean - 2*std), 0) for f in fit_vals]
        
    selection_probabilities = np.array(fit_vals) / sum(fit_vals)
    return selection_probabilities

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
        
        parents.sort()
        couples = parents.reshape(num_couples, 2, 265)

        return couples
    else:
        for i in range(num_couples):
            idx = np.random.choice(len(pop), 2, p=selection_probabilities)
            couples[i] = [pop[idx[0]], pop[idx[1]]]
        
    return couples

def crossover(p1, p2, mutation):
    gene_mask = np.random.rand(len(p1)) <= 0.5

    offspring1 = p1.copy()
    offspring2 = p2.copy()

    offspring1[gene_mask] = p2[gene_mask]
    offspring2[gene_mask] = p1[gene_mask]

    mutation_mask = np.random.rand(len(p1)) <= mutation
    offspring1 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask
    offspring2 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask

    offspring1 = np.clip(offspring1, lb_w, ub_w)
    offspring2 = np.clip(offspring2, lb_w, ub_w)

    return np.array([offspring1, offspring2])
    
def create_offspring(pop, fitness, mutation, probabilities, fps_var, fps_transpose, ranking_var, sampling):
    couples = select_parents(pop, fitness, fps_var, fps_transpose, ranking_var, sampling, probabilities) #default, transpose, windowing, scaling
    total_offspring = np.zeros((0, n_gen))
    for p1, p2 in couples:
        offspring = crossover(p1, p2, mutation)
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
            
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def simulation_gain(env,x):
    f,p,e,t = env.play(pcont=x)
    return p-e
    
def norm(x, pfit_pop):

    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))
        
def print_size_subpops(subpops):
    print('Subpops sizes:')
    for idx, pop in enumerate(subpops):
        print('Subpop {}: {}'.format(idx, len(pop)))
        
def calculate_diversity(population):
    histograms = [w / w.sum() for w in population]
    total_diversity = 0.0

    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            total_diversity += wasserstein_distance(histograms[i], histograms[j])
            
    return total_diversity / (len(population) * (len(population) - 1) / 2)

def count_twins(pop):
    count = 0
    for i in range(len(pop)):
        for j in range (i + 1, len(pop)):
            if np.array_equal(pop[i], pop[j]):
                count += 1
    return count

def round_robin(population, fit_pop, tournament = "random"):
    num_individuals = len(population)
    robin_scores = [1] * num_individuals
    if tournament == "random":
        for i in range(num_individuals):
            for _ in range(10):
                result_i = fit_pop[i]
                j = np.random.randint(0,population.shape[0], 1)[0]
                result_j = fit_pop[j]
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
    selected_indices = []
    
    for _ in range(gen_size):
        total_scores = sum(scores)
        probabilities = scores / total_scores
        
        rand_num = random.uniform(0, 1)
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob >= rand_num:
                selected_indices.append(i)
                scores[i] = 0 
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

def train_specialist_GA(env, mutation, test=False, probabilities_parents="ranking", sampling="sus"):
    ini = time.time()
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
    
    pop = initialize_population(npop)
    fitness_gen = evaluate(env, pop)
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
    best_txt = ''
    pop_prev = np.copy(pop)
    for gen in range(1, gens+1):
        mutation -= 0.015
        offspring = create_offspring(pop, fitness_gen, mutation=mutation, probabilities=probabilities_parents, fps_var="scaling", fps_transpose = 10, ranking_var="linear", sampling=sampling)
        fitness_offspring = evaluate(env, offspring)
        pop = np.vstack((pop, offspring))
        fitness_gen = np.append(fitness_gen, fitness_offspring)
        
        # Find best solution
        best_idx = np.argmax(fitness_gen)
        best = fitness_gen[best_idx]
        best_txt = pop[best_idx]
        
        #survival selection 
        scores = np.array(round_robin(pop, fitness_gen, tournament = "random"))    
        selected = np.array(select_robin(scores, gen_size = npop))
        if best_idx not in selected: 
            selected = np.append(selected[1:], best_idx)
        pop = pop[selected]
        fitness_gen = fitness_gen[selected]
        pop_prev = np.copy(pop)
        std = np.std(fitness_gen)
        mean = np.mean(fitness_gen)
        div = calculate_diversity(pop)
        results.loc[len(results)] = np.array([gen, best, mean, std, div])
        print('Gen: {}, Best: {:.2f}, Mean: {:.2f}, Diversity: {:.2f}'.format(gen, best, mean, div))
        
    best = simulation(env, best_txt)
    
    fim = time.time() # prints total execution time for experiment
    print('Final eval best solution:', best)    
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    return best, results, best_txt


def train_specialist_DGA(env, mutation, migration, n_subpops, sampling='roulette', test=False):
    ini = time.time() 
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
    
    pop = initialize_population(npop)
    subpops = np.array_split(pop, n_subpops, axis=0)
    fitness_subpops = [evaluate(env, x) for x in subpops]
    fitness_gen = np.concatenate(fitness_subpops)
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
    best_txt = ''
    
    for gen in range(1, gens+1):
        for idx, subpop in enumerate(subpops):
            sub_pop_size = len(subpop)
            fitness_subpop = fitness_subpops[idx]
            exchange_information(subpops, fitness_subpops, migration_rate=migration)
            offspring = create_offspring(subpop, fitness_subpop, mutation, probabilities="ranking", fps_var="scaling", fps_transpose = 10, ranking_var="exponential", sampling=sampling)
            fitness_offspring = evaluate(env, offspring)
            cur_pop = np.vstack((subpop, offspring))
            fitness = np.append(fitness_subpop, fitness_offspring)
            best_idx = np.argmax(fitness)
            
            scores = np.array(round_robin(cur_pop, fitness, tournament = "random"))    
            selected = np.array(select_robin(scores, sub_pop_size))
            if not best_idx in selected:
                selected = np.append(selected[1:], best_idx)
            fitness_subpops[idx] = fitness[selected]
            subpops[idx] = cur_pop[selected]
            
        fitness_gen = np.concatenate(fitness_subpops)
        pop = np.concatenate(subpops)
        best_txt = pop[np.argmax(fitness_gen)]
        best = np.max(fitness_gen)
        mean = np.mean(fitness_gen)
        std = np.std(fitness_gen)
        div = calculate_diversity(pop)
        
        results.loc[len(results)] = np.array([gen, best, mean, std, div])
        print('Gen: {}, Best: {:.2f}, Mean: {:.2f}, Diversity: {:.2f}'.format(gen, best, mean, div))
        
    best = simulation(env, best_txt)
    
   
    # np.savetxt(experiment+'/best_DGA_'+ str(enemy) + '_' + str(sol_num) + '.txt',best_txt)
    # div = calculate_diversity(pop)   
    # plot_diversity(results)
    # print('Best final solution: {}'.format(best)) 
    # fim = time.time() # prints total execution time for experiment
    # print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    # print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    
    return best, results, best_txt
    
def hyperparameter_tuning(enemy):
    
    enviroment = Environment(experiment_name=experiment,
                            enemies=[enemy],
                            playermode=playermode,
                            player_controller=player_controller(n_hidden_nodes),
                            enemymode=enemymode,
                            level=2,
                            speed="fastest",
                            visuals=False)
    
    def model_GA(env, hyperparameters):
        return train_specialist_GA(env, mutation=hyperparameters['mutation'], test=True, 
                               probabilities_parents=hyperparameters['probabilities_parents'], sampling=hyperparameters["sampling"])
    
    def model_DGA(env, hyperparameters):
        return train_specialist_DGA(env, hyperparameters['mutation'], hyperparameters['migration'], hyperparameters['n_subpops'], test=True)
    

    def objective(trial):
        # Define hyperparameters to search
        hyperparameters = {
            'mutation': trial.suggest_float('mutation', 0, 0.4),
            'probabilities_parents': 'ranking',
            'sampling': trial.suggest_categorical('sampling', ['roulette', 'sus']),
            'migration': trial.suggest_float('migration', 0, 0.2),
            'n_subpops': trial.suggest_int('n_subpops', 1, 4),
            
        }

        # fitness, results = model_GA(enviroment, enemies, experiment, hyperparameters)
        fitness, results = model_DGA(enviroment, enemies, experiment, hyperparameters)
        return fitness

    print('Starting optimization')
    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_hyperparameters = study.best_params
    best_accuracy = study.best_value

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Accuracy:", best_accuracy)
    res = {'hyperparameters': best_hyperparameters,
            'accuracy': best_accuracy}
    print(res)



experiment = 'generalist_optimization' # name of the experiment
headless = True # True for not using visuals, false otherwise
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test  
    
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment) 

npop = 100
gens = 50 # max number of generations
mutation_GA = 0.333438 # mutation probability
sampling_GA = 'roulette'
mutation_DGA = 0.197246 # mutation probability
sampling_DGA = 'sus'
migration = 0.2
n_subpops = 3

enviroment = Environment(experiment_name=experiment,
                playermode=playermode,
                player_controller=player_controller(n_hidden_nodes),
                enemymode=enemymode,
                multiplemode="yes",
                level=2,
                speed="fastest",
                visuals=False)
enviroment.fitness_single = types.MethodType(fitness_single, enviroment)
enviroment.state_to_log() # checks environment state
n_gen = (enviroment.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector 

enemies = [1,7]
n_runs = 10

enviroment.update_parameter('enemies', enemies)
best_score, results_GA, best_GA = train_specialist_GA(enviroment, mutation=mutation_GA, test=False, 
                                    probabilities_parents='ranking', sampling='sus')
np.savetxt(experiment+'/best_GA_set_test.txt',np.array(best_GA))

# best_score, results_DGA, best_DGA = train_specialist_DGA(enviroment, mutation_DGA, migration, n_subpops, sampling='sus', test=False)
# np.savetxt(experiment+'/best_DGA_set_2.txt',np.array(best_DGA))

# for enemy in enemies:
#     df_DGA = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity', 'solution'])
#     df_GA = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity', 'solution'])
#     best_sols_DGA = []
#     best_sols_GA = []
#     for i in range(n_runs):
#         print('Generating solution {}/{}'.format(i+1,n_runs))
        
#         enviroment.update_parameter('enemies', [enemy])
#         best_score, results_DGA, best_DGA = train_specialist_DGA(enviroment, mutation_DGA, migration, n_subpops, sampling='sus', test=False)
#         best_score, results_GA, best_GA = train_specialist_GA(enviroment, mutation=mutation_GA, test=False, 
#                                     probabilities_parents='ranking', sampling='roulette')

#         gain_DGA = np.mean(np.array([simulation_gain(enviroment, best_DGA) for _ in range(5)]))
#         gain_GA = np.mean(np.array([simulation_gain(enviroment, best_GA) for _ in range(5)]))
#         print('Average gain for DGA solution {}: {}'.format(i+1, gain_DGA))
#         print('Average gain for GA solution {}: {}'.format(i+1, gain_GA))
#         best_sols_DGA.append(best_DGA)
#         best_sols_GA.append(best_GA)          
#         results_DGA['solution'] = i
#         results_GA['solution'] = i
        
#         df_DGA = pd.concat([df_DGA, results_DGA], axis=0, ignore_index=True)
#         df_GA = pd.concat([df_GA, results_GA], axis=0, ignore_index=True)
        
#     np.savetxt(experiment+'/best_DGA_'+ str(enemy) + '.txt',np.array(best_sols_DGA))
#     np.savetxt(experiment+'/best_GA_'+ str(enemy) + '.txt',np.array(best_sols_GA))
            
#     df_DGA.to_csv(experiment+'/results_DGA_' + str(enemy))
#     df_GA.to_csv(experiment+'/results_GA_' + str(enemy))
        
        
# import cProfile
# cProfile.run("train_specialist_GA(enviroment, mutation_GA, False, 'ranking', 'sus')", sort="cumulative")
    
    