import numpy as np


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
n_subpops = 

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
subpops = np.array_split(pop, n_subpops, axis=0)
print_size_subpops(subpops)

exchange_information(subpops)
print_size_subpops(subpops)