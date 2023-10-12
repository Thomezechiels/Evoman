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
    
    vfitness = np.mean(vfitness)
    vplayerlife = np.mean(vplayerlife)
    venemylife = np.mean(venemylife)
    vtime = np.mean(vtime)

    return vfitness, vplayerlife, venemylife, vtime

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

def is_within_threshold(individual, group, threshold):
    for member in group:
        dist = calculate_euclidian(individual, member)
        # print('Eculidian dist:', dist)
        if dist > threshold:
            return False
    return True

def cluster_individuals(pop, threshold_within_group, threshold_to_group):
    groups = []

    for i in pop:
        added_to_group = False
        
        for group in groups:
            
            avg_similarity = sum(calculate_euclidian(i, j) for j in group) / len(group)
            # Check if the individual can be added to the group without exceeding the threshold for individual distances
            if is_within_threshold(i, group, threshold_within_group) and avg_similarity <= threshold_to_group:
                group.append(i)
                added_to_group = True
                break

        # If the individual doesn't fit in any existing group, create a new group
        if not added_to_group:
            groups.append([i])

    return groups

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))

# def evaluate(env, x):
#     groups = cluster_individuals(x, 8, 8)
#     individuals = np.concatenate(groups)
#     fit_vals = np.zeros(0)
#     for g in groups:
#         fitness_group = np.full(len(g), simulation(env, g[0]))
#         fit_vals = np.append(fit_vals, fitness_group)
#     print('{} individuals -> {} groups'.format(len(x), len(groups)))
#     org = np.array(list(map(lambda y: simulation(env, y), individuals)))
#     diff = org-fit_vals
#     formatted_list = ["{:.3f}".format(num) for num in diff if abs(num) > 3]
#     print('Difference fit vals:', formatted_list)
#     return individuals, fit_vals

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

        # Initialize pointers for best and worst parents
        # best_index = 0
        # worst_index = len(parents) - 1

        # # Loop through and alternate between best and worst parents
        # for i in range(len(parents)):
        #     if i % 2 == 0:
        #         # Even index, insert the best parent
        #         sorted_parents[i] = parents[sorted_indices[best_index]]
        #         best_index += 1
        #     else:
        #         # Odd index, insert the worst parent
        #         sorted_parents[i] = parents[sorted_indices[worst_index]]
                # worst_index -= 1

        return parents.reshape(num_couples, 2, 265)
    else:
        for i in range(num_couples):
            idx = np.random.choice(len(pop), 2, p=selection_probabilities)
            couples[i] = [pop[idx[0]], pop[idx[1]]]
        
    return couples

def crossover(p1, p2, n_offspring, mutation):
    offspring = []

    for _ in range(n_offspring):
        gene_mask = np.random.rand(len(p1)) <= 0.5

        child = p1.copy()
        child[gene_mask] = p2[gene_mask]

        mutation_mask = np.random.rand(len(p1)) <= mutation
        child += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask

        child = np.clip(child, lb_w, ub_w)
        offspring.append(child)

    return np.array(offspring)

import numpy as np

def one_point_crossover(p1, p2, n=2, mutation=0.1):
    # Ensure both parents have the same length
    if len(p1) != len(p2):
        raise ValueError("Parents must have the same length for one-point crossover")

    # Initialize a list to store offspring
    offspring = []

    for _ in range(round(n/2)):
        # Randomly choose a crossover point
        crossover_point = np.random.randint(1, len(p1))

        # Create two empty offspring with the same length as parents
        offspring1 = np.empty_like(p1)
        offspring2 = np.empty_like(p2)

        # Perform one-point crossover
        offspring1[:crossover_point] = p1[:crossover_point]
        offspring1[crossover_point:] = p2[crossover_point:]
        
        offspring2[:crossover_point] = p2[:crossover_point]
        offspring2[crossover_point:] = p1[crossover_point:]

        mutation_mask_1 = np.random.rand(len(p1)) <= mutation
        offspring1 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask_1
        mutation_mask_2 = np.random.rand(len(p1)) <= mutation
        offspring2 += np.random.normal(0, 0.5, size=len(p1)) * mutation_mask_2
        
        offspring1 = np.clip(offspring1, lb_w, ub_w)
        offspring2 = np.clip(offspring2, lb_w, ub_w)
        
        offspring.append(offspring1)
        offspring.append(offspring2)

    return offspring


import numpy as np

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
        # offspring = one_point_crossover(p1, p2, 2, mutation)
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
        
def print_size_subpops(subpops):
    print('Subpops sizes:')
    for idx, pop in enumerate(subpops):
        print('Subpop {}: {}'.format(idx, len(pop)))
        
def calculate_wasserstein(i1, i2):
    i1 = i1 / i1.sum()
    i2 = i2 / i2.sum()
    
    return wasserstein_distance(i1, i2)

def calculate_euclidian(i1, i2):
    return np.linalg.norm(i1 - i2)

def calculate_diversity(population):
    histograms = [w / w.sum() for w in population]
    total_diversity = 0.0

    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            total_diversity += wasserstein_distance(histograms[i], histograms[j])
            
    return total_diversity / (len(population) * (len(population) - 1) / 2)

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

def train_specialist_GA(env, mutation, test=False, probabilities_parents="ranking", sampling="sus", n_crossover=2, sigma_share=15, pop_given=[]):
    ini = time.time()
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
    
    
    pop = initialize_population(npop)
    if len(pop_given) > 0:
        pop = pop_given
    fitness_gen = evaluate(env, pop)
    fitness_gen_og = fitness_gen.copy()
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
    best_txt = ''
    best = -1000
    for gen in range(1, 50+1):
        mutation -= 0.001
        enemies_selected = list(np.random.choice(enemies, 2, replace=False))
        env.update_parameter('enemies', enemies_selected)
        offspring = create_offspring(pop, fitness_gen, mutation=mutation, probabilities=probabilities_parents, fps_var="scaling", 
                                     fps_transpose = 10, ranking_var="exponential", sampling=sampling, sorted=(gen%2==0), 
                                     n_crossover=n_crossover, n_offspring = 4)
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
        num_top = round(npop * 0.1)
        top_indices = np.argsort(fitness_gen_og)[::-1][:num_top]
        
        # top_individuals = pop[top_indices]
        
        mask = np.ones_like(np.zeros(len(pop)), dtype=bool)
        mask[top_indices] = False

        # Get the elements not in the top 4 indices
        selection_pop = pop[mask]
        
        #Share fitness
        fitness_gen = calculate_sharing_fitness(pop, fitness_gen_og, sigma_share)
        # fitness_gen = fitness_gen_og
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
        div = calculate_diversity(pop)
        results.loc[len(results)] = np.array([gen, best, mean, std, div])
        print('Gen: {}, Best: {:.2f}, Mean: {:.2f}, Diversity: {:.2f}'.format(gen, best, mean, div))
        
    best = simulation(env, best_txt)
    
    fim = time.time() # prints total execution time for experiment
    print('Final eval best solution:', best)    
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')
    return best, results, best_txt


def train_specialist_DGA(env, mutation, migration, n_subpops, sampling='roulette', test=False, n_crossover=2, sigma_share=15, stacked_GA=False):
    results = pd.DataFrame(columns=['gen', 'best', 'mean', 'std', 'diversity'])
    if stacked_GA:
        n_subpops = len(enemies)
    pop = initialize_population(npop)
    subpops = np.array_split(pop, n_subpops, axis=0)
    fitness_subpops = []
    if stacked_GA:
        env.update_parameter('multiplemode', 'no')
        for idx, subpop in enumerate(subpops):
            env.update_parameter('enemies', [enemies[idx]])
            fitness_subpops.append(evaluate(env, subpop))
    else:
        fitness_subpops = [evaluate(env, x) for x in subpops]
    fitness_gen = np.concatenate(fitness_subpops)
    results.loc[len(results)] = np.array([0, np.max(fitness_gen), np.mean(fitness_gen), np.std(fitness_gen), calculate_diversity(pop)])
    
    for gen in range(1, gens+1):
        if gen % 5 == 0:
            exchange_information(subpops, fitness_subpops, migration_rate=migration)
        for idx, subpop in enumerate(subpops):
            if stacked_GA:
                env.update_parameter('enemies', [enemies[idx]])
            sub_pop_size = len(subpop)
            fitness_subpop = fitness_subpops[idx]
            best_ind = subpop[np.argmax(fitness_subpop)]
            best = np.max(fitness_subpop)
            offspring = create_offspring(subpop, fitness_subpop, mutation=mutation, probabilities="ranking", fps_var="scaling", 
                                         fps_transpose = 10, ranking_var="exponential", sampling=sampling, sorted=(gen%2==0), 
                                         n_crossover=n_crossover, n_offspring=4)
            fitness_offspring = evaluate(env, offspring)
            best_offspring = np.max(fitness_offspring)
            best_idx = 0
            if best_offspring > best:
                # Get best solution
                best_idx = np.argmax(fitness_offspring)
                # Make population children only
                subpop = offspring
                fitness_subpop = fitness_offspring
            else:
                best_idx = len(offspring)
                subpop = np.vstack([offspring, best_ind])
                fitness_subpop = np.append(fitness_offspring, best)
            
            #survival selection 
            
            
            if gen == gens and stacked_GA:
                top_indices = np.argsort(fitness_subpop)[::-1][:sub_pop_size]
                subpops[idx] = subpop[top_indices]
                top_scores = fitness_subpop[top_indices]
                print('Top scores subpop {}\n'.format(idx), top_scores)
            else:    
                num_top = round(sub_pop_size * 0.20)
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
                # if not best_idx in selected:
                # selected = np.append(selected[1:], best_idx)
                fitness_subpops[idx] = fitness_subpop[selected]
                subpops[idx] = subpop[selected]
            
        fitness_gen = np.concatenate(fitness_subpops)
        pop = np.concatenate(subpops)
        best_idx = np.argmax(fitness_gen)
        best_DGA = pop[best_idx]
        best = np.max(fitness_gen)
        mean = np.mean(fitness_gen)
        std = np.std(fitness_gen)
        div = calculate_diversity(pop)
        
        results.loc[len(results)] = np.array([gen, best, mean, std, div])
        print('Gen: {}, Best: {:.2f}, Mean: {:.2f}, Diversity: {:.2f}'.format(gen, best, mean, div))
        if stacked_GA:
            for fit, enemy in zip(fitness_subpops, enemies):
                print('Enemy {} best score of {:.2f}'.format(enemy, np.max(fit)))
        
    if stacked_GA:  
        env.update_parameter('multiplemode', 'yes')    
        env.update_parameter('enemies', enemies)    
        return train_specialist_GA(env, mutation=mutation_GA, test=False, 
                                    probabilities_parents='ranking', sampling='sus', n_crossover = n_crossover_GA, 
                                        sigma_share=sigma_share_GA, pop_given=pop)    

    best = simulation(env, best_DGA)
    return best, results, best_DGA


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
# enviroment.state_to_log() # checks environment state
n_gen = (enviroment.get_num_sensors()+1)*n_hidden_nodes + (n_hidden_nodes+1)*5 #size of weight vector 
enemies = [3,6,7,8]
n_runs = 10
enviroment.update_parameter('enemies', enemies)

npop = 60
gens = 1 # max number of generations

migration = 0.1
n_subpops = 4

n_crossover_GA = 6
sigma_share_GA = 0.05
mutation_GA = 0.15 # mutation probability

n_crossover_DGA = 3
sigma_share_DGA = 12
mutation_DGA = 0.197246 # mutation probability

# best_score, results_GA, best_GA = train_specialist_GA(enviroment, mutation=mutation_GA, test=False, 
#                                     probabilities_parents='ranking', sampling='sus', n_crossover = n_crossover_GA, sigma_share=sigma_share_GA)
# np.savetxt(experiment+'/best_GA_set_test_6.txt',np.array(best_GA))

best_score, results_DGA, best_DGA = train_specialist_DGA(enviroment, mutation_DGA, migration, n_subpops, sampling='sus', test=False, 
                                                         n_crossover=n_crossover_DGA, sigma_share=sigma_share_DGA, stacked_GA=True)
np.savetxt(experiment+'/best_DGA_set_test_4.txt',np.array(best_DGA))

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


# def objective(trial):
#     mutation = trial.suggest_float('mutation', 0, 0.15)
#     # migration = trial.suggest_float('migration', 0, 0.2)
#     n_point_crossover = trial.suggest_int('n_point_crossover', 3, 6)
#     sigma_share = trial.suggest_int('sigma_share', 3, 20)
#     best_score, results_GA, best_GA = train_specialist_GA(enviroment, mutation=mutation, test=False, 
#                                 probabilities_parents='ranking', sampling='sus', n_crossover = n_point_crossover, sigma_share=sigma_share)
    
#     if best_score > 70:
#         np.savetxt(experiment+'/best_GA_Optuna_'+str(trial.number)+'.txt',np.array(best_GA))
#     # fitness, results = model_DGA(enviroment, enemies, experiment, hyperparameters)
    
#     return best_score


# print('Starting optimization')
# # Create Optuna study
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100, n_jobs=1, gc_after_trial =True, show_progress_bar=True)

# # Get the best hyperparameters
# best_hyperparameters = study.best_params
# best_accuracy = study.best_value

# print("Best Hyperparameters:", best_hyperparameters)
# print("Best Accuracy:", best_accuracy)
# res = {'hyperparameters': best_hyperparameters,
#         'accuracy': best_accuracy}
# print(res)
    
    