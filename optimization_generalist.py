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
        return 0.6*(100 - self.get_enemylife()) + 0.4*self.get_playerlife() - 0.0 * np.log(self.get_time())
        # return self.get_playerlife() - 2 * np.log(self.get_time()) #self.get_enemylife()
        
# repeats run for every enemy in list
def multiple(self,pcont,econt):

    vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
    for e in self.enemies:

        fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
        vfitness.append(fitness)
        vplayerlife.append(playerlife)
        venemylife.append(enemylife)
        vtime.append(time)
        
    # t_fitness = sum([p - e - np.log(t) for p, e, t in zip(vplayerlife, venemylife, vtime)]) / len(vplayerlife)

    # vfitness = self.cons_multi(np.array(vfitness))
    # vplayerlife = self.cons_multi(np.array(vplayerlife))
    # venemylife = self.cons_multi(np.array(venemylife))
    # vtime = self.cons_multi(np.array(vtime))
    
    # vfitness = np.mean(vfitness)
    # vplayerlife = np.mean(vplayerlife)
    # venemylife = np.mean(venemylife)
    # vtime = np.mean(vtime)

    return vfitness, vplayerlife, venemylife, vtime

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)    
    return np.mean(f)

def simulation_gain(env,x):
    f,p,e,t = env.play(pcont=x)
    gain = sum(p) - sum(e)
    return gain
    
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
    

def select_parents(pop, fit_vals, fps_var="default", fps_transpose=10, ranking_var="exponential", sampling="sus", probabilities="fps", sorted=True):
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
    # print(adjusted_fitness)
    if probabilities == 'fps':
        selection_probabilities = selection_fps(adjusted_fitness, fps_var, fps_transpose)
    elif probabilities == 'ranking':
        selection_probabilities = selection_ranking(pop, adjusted_fitness, ranking_var='linear')
            
    if sampling == 'sus':
        num_parents = num_couples * 2  # We select twice the number of parents since each couple has 2 parents
        parents = np.zeros((num_parents, n_gen))
        parents_fitness = np.zeros(num_parents)
        current_member = i = 0
        r = np.random.uniform(0, 1 / num_parents)
        cumulative_probabilities = np.cumsum(selection_probabilities)
        
        while current_member < num_parents:
            while r <= cumulative_probabilities[i]:
                parents[current_member] = pop[i]
                parents_fitness[current_member] = fit_vals[i]
                r += 1 / num_parents
                current_member += 1
            i += 1
        if sorted:
            sorted_indices = np.argsort(parents_fitness)

            # Create an array for the sorted parents
            parents = parents[sorted_indices]

        return parents.reshape(num_couples, 2, 265)
    else:
        for i in range(num_couples):
            idx = np.random.choice(len(pop), 2, p=selection_probabilities)
            couples[i] = [pop[idx[0]], pop[idx[1]]]
        
    return couples

def n_point_crossover(p1, p2, n=2, n_points=2, mutation=0.1):
    if len(p1) != len(p2):
        raise ValueError("Parents must have the same length for n-point crossover")

    offspring = []

    for _ in range(round(n/2)):
        # Generate n unique crossover points
        crossover_points = sorted(np.random.choice(len(p1), n_points, replace=False))

        # Create two empty offspring with the same length as parents
        offspring1 = np.empty_like(p1)
        offspring2 = np.empty_like(p2)

        # Perform n-point crossover
        for i in range(n_points + 1):
            start = 0 if i == 0 else crossover_points[i - 1]
            end = len(p1) if i == n_points else crossover_points[i]
            
            if i % 2 == 0:
                offspring1[start:end] = p1[start:end]
                offspring2[start:end] = p2[start:end]
            else:
                offspring1[start:end] = p2[start:end]
                offspring2[start:end] = p1[start:end]

        mutation_mask_1 = np.random.rand(len(p1)) <= mutation
        offspring1 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask_1
        mutation_mask_2 = np.random.rand(len(p1)) <= mutation
        offspring2 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask_2
        
        offspring1 = np.clip(offspring1, lb_w, ub_w)
        offspring2 = np.clip(offspring2, lb_w, ub_w)
        
        offspring.append(offspring1)
        offspring.append(offspring2)
        
    return offspring

    
def create_offspring(pop, fitness, mutation, probabilities, fps_var, fps_transpose, ranking_var, sampling, sorted, n_crossover, n_offspring):
    couples = select_parents(pop, fitness, fps_var, fps_transpose, ranking_var, sampling, probabilities, sorted) #default, transpose, windowing, scaling
    total_offspring = np.zeros((0, n_gen))
    for p1, p2 in couples:
        offspring = n_point_crossover(p1, p2, n_offspring, n_points=n_crossover, mutation=mutation)
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
        # subpops[idx] = np.delete(subpop, migrants_idx, axis=0)
        # fitness_subpops[idx] = np.delete(fitness_subpops[idx], migrants_idx, axis=0)
        
    for idx, subpop in enumerate(subpops):
        if len(migrations[idx]) > 0:
            subpops[idx] = np.append(subpop, migrations[idx], axis=0)
            fitness_subpops[idx] = np.append(fitness_subpops[idx], migrations_fitness[idx], axis=0)
        
def calculate_wasserstein(i1, i2):
    i1 = i1 / i1.sum()
    i2 = i2 / i2.sum()
    
    return wasserstein_distance(i1, i2)

def calculate_euclidian(i1, i2):
    return np.linalg.norm(i1 - i2)

def calculate_sharing_fitness(pop, fitness, sigma_share):
    sharing_fitness = fitness.copy()

    for idx, (i1, fit1) in enumerate(zip(pop, fitness)):
        denominator = 1
        for i2, fit2 in zip(pop, fitness):
            if not np.array_equiv(i1,i2):
                distance_ij = calculate_euclidian(i1, i2)
                if distance_ij < sigma_share:
                    denominator += (1-(distance_ij/sigma_share))
        sharing_fitness[idx] = fit1/denominator
        # print('Reduced fitness of {} by: {:.2f}%'.format(idx, (((1-denominator)/denominator)*100)))
    return sharing_fitness

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

def train_specialist_GA(env, mutation, probabilities_parents="ranking", sampling="sus", n_crossover=2, 
                        sigma_share=15, pop_given=[], elitism_rate=0.1, enemies_to_train=3):
    best_txt = []
    best = -1000
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std'])
    
    pop = initialize_population(npop)
    if len(pop_given) > 0:
        pop = pop_given

    fitness_gen = evaluate(env, pop)
    fitness_gen_og = fitness_gen.copy()
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen)])
    mutation_run = mutation
    for gen in range(1, gens_GA+1):
        mutation_run -= 0.003
        enemies_selected = list(np.random.choice(enemies, enemies_to_train, replace=False))
        env.update_parameter('enemies', enemies_selected)
        offspring = create_offspring(pop, fitness_gen, mutation=mutation_run, probabilities=probabilities_parents, fps_var="scaling", 
                                     fps_transpose = 10, ranking_var="exponential", sampling=sampling, sorted=(gen%2==0), 
                                     n_crossover=n_crossover, n_offspring = 6)
        fitness_offspring = evaluate(enviroment, offspring)
        best_offspring = np.max(fitness_offspring)
        if best_offspring > best:
            # Get best solution
            best_idx = np.argmax(fitness_offspring)
            best = fitness_offspring[best_idx]
            best_txt = offspring[best_idx]
            # Make population children only
            pop = offspring
            fitness_gen_og = fitness_offspring
        else:
            best_idx = len(offspring)
            pop = np.vstack([offspring, best_txt])
            fitness_gen_og = np.append(fitness_offspring, best)   
            
        #survival selection 
        num_top = round(npop * elitism_rate)
        top_indices = np.argsort(fitness_gen_og)[::-1][:num_top]
        
        # top_individuals = pop[top_indices]
        
        mask = np.ones_like(np.zeros(len(pop)), dtype=bool)
        mask[top_indices] = False

        # Get the elements not in the top 4 indices
        selection_pop = pop[mask]
        
        #Share fitness
        # fitness_gen = calculate_sharing_fitness(pop, fitness_gen_og, sigma_share)
        fitness_gen = fitness_gen_og
        selection_fitness = fitness_gen[mask]
        scores = np.array(round_robin(selection_pop, selection_fitness, tournament = "random")) 
        selected = np.array(select_robin(scores, gen_size = npop-num_top))
       
        # if not best_idx in selected:
        selected = np.concatenate((selected, top_indices))
        pop = pop[selected]
        fitness_gen = fitness_gen[selected]
        fitness_gen_og = fitness_gen_og[selected]
        
        std = np.std(fitness_gen_og)
        mean = np.mean(fitness_gen_og)
        results.loc[len(results)] = np.array([gen, best, mean, std])
        print('Gen: {}, Best: {:.2f}, Mean: {:.2f}, Std: {:.2f}'.format(gen, best, mean, std))

        
    return np.max(fitness_gen_og), results, pop[np.argmax(fitness_gen_og)]


def train_specialist_DGA(env, mutation_DGA, migration, mutation_GA, sampling_GA, probabilities_parents_GA, n_point_crossover_GA, 
                         n_point_crossover_DGA, elitism_rate, enemies_to_train, stacked_GA=True):
    pop = []
    results = pd.DataFrame(columns=['gen', 'enemy', 'best', 'mean'])
    if stacked_GA:
        
        n_subpops = len(enemies)
        pop = initialize_population(npop_DGA)
        subpops = np.array_split(pop, n_subpops, axis=0)
        fitness_subpops = []
        env.update_parameter('multiplemode', 'no')
        for idx, subpop in enumerate(subpops):
            env.update_parameter('enemies', [enemies[idx]])
            fitness_subpops.append(evaluate(env, subpop))
        fitness_gen = np.concatenate(fitness_subpops)
        for gen in range(1, gens_DGA+1):
            if gen % 5 == 0:
                exchange_information(subpops, fitness_subpops, migration_rate=migration)
            for idx, subpop in enumerate(subpops):
                env.update_parameter('enemies', [enemies[idx]])
                sub_pop_size = len(subpop)
                fitness_subpop = fitness_subpops[idx]
                best_ind = subpop[np.argmax(fitness_subpop)]
                best = np.max(fitness_subpop)
                offspring = create_offspring(subpop, fitness_subpop, mutation=mutation_DGA, probabilities="ranking", fps_var="scaling", 
                                            fps_transpose = 10, ranking_var="exponential", sampling=sampling_GA, sorted=False, 
                                            n_crossover=n_point_crossover_DGA, n_offspring=4)
                fitness_offspring = evaluate(env, offspring)
                best_offspring = np.max(fitness_offspring)
               
                if best_offspring > best:
                    # Get best solution
                    subpop = offspring
                    fitness_subpop = fitness_offspring
                else:
                    subpop = np.vstack([offspring, best_ind])
                    fitness_subpop = np.append(fitness_offspring, best)
                
                if gen == gens_DGA + 1:
                    new_pop_reduction = round((npop / npop_DGA) * sub_pop_size)
                    top_indices = np.argsort(fitness_subpop)[::-1][:new_pop_reduction]
                    subpops[idx] = subpop[top_indices]
                else:    
                    num_top = round(sub_pop_size * elitism_rate)
                    top_indices = np.argsort(fitness_subpop)[::-1][:num_top]
                    mask = np.ones_like(np.zeros(len(subpop)), dtype=bool)
                    mask[top_indices] = False

                    # Get the elements not in the top 4 indices
                    selection_subpop = subpop[mask]
                    
                    #Share fitness
                    # fitness_gen = calculate_sharing_fitness(pop, fitness_gen_og, sigma_share)
                    selection_fitness = fitness_subpop[mask]
                
                    scores = np.array(round_robin(selection_subpop, selection_fitness, tournament = "random"))    
                    selected = np.array(select_robin(scores, sub_pop_size-num_top))
                    selected = np.concatenate((selected, top_indices))
                    fitness_subpops[idx] = fitness_subpop[selected]
                    subpops[idx] = subpop[selected]
            
            for idx, fitness_subpop in enumerate(fitness_subpops):
                best_subpop = np.max(fitness_subpop)
                mean_subpop = np.mean(fitness_subpop)
                results.loc[len(results)] = np.array([gen, enemies[idx], best_subpop, mean_subpop])  
                print('Gen: {}, Enemy: {}, Best: {:.2f}, Mean: {:.2f}'.format(gen, enemies[idx], best_subpop, mean_subpop))
            fitness_gen = np.concatenate(fitness_subpops)
            pop = np.concatenate(subpops)
            best = np.max(fitness_gen)

    return results  
    # env.update_parameter('multiplemode', 'yes')    
    # env.update_parameter('enemies', enemies)    
    # return train_specialist_GA(env, mutation=mutation_GA, 
    #                             probabilities_parents=probabilities_parents_GA, sampling=sampling_GA, n_crossover = n_point_crossover_GA, 
    #                                 sigma_share=10, pop_given=pop, elitism_rate=elitism_rate, enemies_to_train=enemies_to_train)    


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

enviroment = Environment(experiment_name=experiment,
                playermode=playermode,
                player_controller=player_controller(n_hidden_nodes),
                enemymode=enemymode,
                multiplemode="yes",
                level=2,
                logs = "off",
                speed="fastest",
                visuals=False)

enviroment.fitness_single = types.MethodType(fitness_single, enviroment)
enviroment.multiple = types.MethodType(multiple, enviroment)
n_gen = (enviroment.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector 

npop_DGA = 100
npop = 100
gens_DGA = 30
gens_GA = 50

migration = 0.1
n_subpops = 4

n_point_crossover_DGA = 3
sigma_share_DGA = 12
mutation_DGA = 0.197246 # mutation probability


mutation_GA = 0.1530705829472316
probabilities_parents_GA = 'ranking'
sampling_GA = 'sus'
n_point_crossover_GA = 4 
elitism_rate = 0.16900216930076012
enemies_to_train = 4
enemy_sets = {
                '1': [1,2,5,7], 
                '2': [3,4,6,8],
            }
enemies = []
nruns = 10

for set_num, enemy_set in enemy_sets.items():
    enemies = enemy_set
    df_results = pd.DataFrame(columns=['gen', 'enemy', 'best', 'mean', 'solution'])
    best_sols_DGA = []
    for i in range(nruns):
        print('Generating solution {}/{}'.format(i+1,nruns))
        results = train_specialist_DGA(enviroment, mutation_DGA, migration, mutation_GA, sampling_GA, 
                                                                probabilities_parents_GA, n_point_crossover_GA, n_point_crossover_DGA, 
                                                                elitism_rate, enemies_to_train, stacked_GA=True)
        enviroment.update_parameter('enemies', enemies)
        results['solution'] = i
        
        df_results = pd.concat([df_results, results], axis=0, ignore_index=True)
            
    df_results.to_csv(experiment+'/results_DGA_'+str(set_num)+'.csv')

# for set_num, enemy_set in enemy_sets.items():
#     enemies = enemy_set
#     df_DGA = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'solution'])
#     df_GA = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'solution'])
#     best_sols_DGA = []
#     best_sols_GA = []
#     for i in range(nruns):
#         print('Generating solution {}/{}'.format(i+1,nruns))
#         best_score, results_DGA, best_DGA = train_specialist_DGA(enviroment, mutation_DGA, migration, mutation_GA, sampling_GA, 
#                                                                 probabilities_parents_GA, n_point_crossover_GA, n_point_crossover_DGA, 
#                                                                 elitism_rate, enemies_to_train, stacked_GA=True)
#         best_score, results_GA, best_GA = train_specialist_DGA(enviroment, mutation_DGA, migration, mutation_GA, sampling_GA, 
#                                                                 probabilities_parents_GA, n_point_crossover_GA, n_point_crossover_DGA, 
#                                                                 elitism_rate, enemies_to_train, stacked_GA=False)
#         enviroment.update_parameter('enemies', enemies)
#         gain_DGA = simulation_gain(enviroment, best_DGA)
#         gain_GA = simulation_gain(enviroment, best_GA)
#         print('Gain for guided initialization solution {}: {}'.format(i+1, gain_DGA))
#         print('Gain for random solution {}: {}'.format(i+1, gain_GA))
#         best_sols_DGA.append(best_DGA)
#         best_sols_GA.append(best_GA)          
#         results_DGA['solution'] = i
#         results_GA['solution'] = i
        
#         df_DGA = pd.concat([df_DGA, results_DGA], axis=0, ignore_index=True)
#         df_GA = pd.concat([df_GA, results_GA], axis=0, ignore_index=True)
        
#     np.savetxt(experiment+'/best_GI_set_'+str(set_num)+'.txt',np.array(best_sols_DGA))
#     np.savetxt(experiment+'/best_RI_runs_'+str(set_num)+'.txt',np.array(best_sols_GA))
            
#     df_DGA.to_csv(experiment+'/results_GI_'+str(set_num)+'.csv')
#     df_GA.to_csv(experiment+'/results_RI_'+str(set_num)+'.csv')

# import cProfile
# cProfile.run("train_specialist_GA(enviroment, mutation_GA, False, 'ranking', 'sus')", sort="cumulative")


# import contextlib

# def objective(trial):
#     # migration = trial.suggest_float('migration', 0, 0.3)
#     mutation_GA = trial.suggest_float('mutation_GA', 0, 0.15)
#     # mutation_DGA = trial.suggest_float('mutation_DGA', 0.1, 0.3)
#     n_point_crossover_GA = trial.suggest_int('n_point_crossover_GA', 2, 6)
#     # n_point_crossover_DGA = trial.suggest_int('n_point_crossover_DGA', 1, 6)
#     sampling_GA = trial.suggest_categorical('sampling_GA', ['sus', 'roulette'])
#     # probabilities_parents_GA = trial.suggest_categorical('probabilities_parents_GA', ['ranking', 'fps'])
#     probabilities_parents_GA = 'ranking'
#     elitism_rate = trial.suggest_float('elitism_rate', 0, 0.4)
#     enemies_to_train = trial.suggest_int('enemies_to_train', 2, 4)
    
#     # best_score, results_DGA, best_DGA = train_specialist_DGA(enviroment, mutation_DGA, migration, mutation_GA, sampling_GA, 
#     #                                                          probabilities_parents_GA, n_point_crossover_GA, n_point_crossover_DGA, 
#     #                                                          elitism_rate, enemies_to_train, stacked_GA=True)
#     best_score, results_GA, best_GA = train_specialist_GA(enviroment, mutation_GA, probabilities_parents_GA, sampling_GA, n_point_crossover_GA, 
#                         sigma_share=15, pop_given=[], elitism_rate=elitism_rate, enemies_to_train=enemies_to_train)
#     if best_score > 50:
#         np.savetxt(experiment+'/optuna_results/best_'+str(trial.number)+'.txt', np.array(best_GA))

#         text = '{:.2f} on trial {} with params: '.format(best_score, trial.number) + str(trial.params) + '\n'
#         filepath = experiment+'/optuna_results/parameters_log.txt'
#         with contextlib.suppress(FileNotFoundError):
#             with open(filepath, 'a') as file:
#                 file.write(text)
    
#     return best_score

# print('Starting optimization')
# if not os.path.exists(experiment + '/optuna_results'):
#     os.makedirs(experiment + '/optuna_results') 
# # Create Optuna study

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100, gc_after_trial=True)

# # Get the best hyperparameters
# best_hyperparameters = study.best_params
# best_accuracy = study.best_value

# print("Best Hyperparameters:", best_hyperparameters)
# print("Best Accuracy:", best_accuracy)
# res = {'hyperparameters': best_hyperparameters,
#         'accuracy': best_accuracy}
# print(res)
    
    