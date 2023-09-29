import numpy as np
import random


experiment = 'GA_optimization' # name of the experiment
headless = True # True for not using visuals, false otherwise
enemies = [1,2,7]
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test
npop = 157 # size of population
gens = 50 # max number of generations
mutation = 0.1 # mutation probability 
n_gen = 265
n_subpops = 4

def initialize_population(n):
    return np.random.uniform(lb_w, ub_w, (n, n_gen))

def exchange_information(subpops, migration_rate=0.1):
    n_subpops = len(subpops)
    n_individuals = np.array([len(s) for s in subpops])
    n_to_migrate = (migration_rate * n_individuals).astype(int)
    migrations = [[] for _ in range(n_subpops)]
    # Perform migration from each subpopulation to others
    
    
    for idx, subpop in enumerate(subpops):
        migrants_idx = np.random.choice(len(subpop), n_to_migrate[idx], replace=False)
        migrants = subpop[migrants_idx]
        
        for i, migrant in enumerate(migrants):
            i_adjusted = (i + idx) % n_subpops
            migrations[i_adjusted].append(migrant)
        subpops[idx] = np.delete(subpop, migrants_idx, axis=0)
        
    for idx, subpop in enumerate(subpops):
        if len(migrations[idx]) > 0:
            subpops[idx] = np.append(subpop, migrations[idx], axis=0)
        
def print_size_subpops(subpops):
    print('Subpops sizes:')
    for idx, pop in enumerate(subpops):
        print('Subpop {}: {}'.format(idx, len(pop)))
        
pop = initialize_population(npop)


def crossover(p1, p2):
    offspring1 = []
    offspring2 = []

    for gene1, gene2 in zip(p1, p2):
        if random.random() < 0.5:
            offspring1.append(gene1)
            offspring2.append(gene2)
        else:
            offspring1.append(gene2)
            offspring2.append(gene1)

        # mutation
        for i in range(0,len(offspring1)):
            if np.random.uniform(0, 1)<=mutation:
                offspring1[i] = offspring1[i]+np.random.normal(0, 0.5)
                offspring2[i] = offspring2[i]+np.random.normal(0, 0.5)

    return np.array([offspring1, offspring2])


