import sys
from evoman.environment import Environment
from controller import player_controller

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import glob, os











df_GA_1 = pd.read_csv('results_csv/results_GA_1.csv')
df_GA_2 = pd.read_csv('results_csv/results_GA_2.csv')
df_GA_7 = pd.read_csv('results_csv/results_GA_7.csv')
df_DGA_1 = pd.read_csv('results_csv/results_DGA_1.csv')
df_DGA_2 = pd.read_csv('results_csv/results_DGA_2.csv')
df_DGA_7 = pd.read_csv('results_csv/results_DGA_7.csv')

list_files = [df_GA_1, df_GA_2, df_GA_7, df_DGA_1, df_DGA_2, df_DGA_7]

list_df = []
for df in list_files:
    df_avg = df[['gen', 'best', 'mean']].groupby(['gen']).mean()
    df_avg[['best_std', 'mean_std']] = df[['gen', 'best', 'mean']].groupby(['gen']).std()
    df_avg['lower_best'] = df_avg['best'] - df_avg['best_std']
    df_avg['upper_best'] = df_avg['best'] + df_avg['best_std']
    df_avg['lower_mean'] = df_avg['mean'] - df_avg['mean_std']
    df_avg['upper_mean'] = df_avg['mean'] + df_avg['mean_std']
    list_df.append(df_avg)


fig, axs = plt.subplots(1, 3)

for df in range(3):
    axs[df].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df].set_title('Enemy {}'.format(df))
    axs[df].legend(['Best', 'Mean'])

for ax in axs.flat:
    ax.set(xlabel='generations', ylabel='fitness score')

for ax in axs.flat:
    ax.label_outer()
plt.show()



fig, axs = plt.subplots(1, 3)

for df in range(2, 5):
    axs[df-2].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df-2].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df-2].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df-2].set_title('Enemy {}'.format(df))
    axs[df-2].legend(['Best', 'Mean'])

for ax in axs.flat:
    ax.set(xlabel='generations', ylabel='fitness score')

for ax in axs.flat:
    ax.label_outer()
plt.show()




experiment = 'GA_optimization' # name of the experiment
headless = True # True for not using visuals, false otherwise
enemies = [1,2,7]
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment)
    




def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e,t

directory = 'best_scores'
dir = [x[0] for x in os.walk(directory)]


print(dir)
dir = dir[1:]

tuple_list = []
for folder in dir:
    print('folder', folder)
    
    list_ = list()

    print(folder[-1])
    enemy = folder[-1]
    print('enemyL ', enemy)
    ENV = Environment(experiment_name=experiment,
                    enemies=enemy,
                    playermode=playermode,
                    player_controller=player_controller(n_hidden_nodes),
                    enemymode=enemymode,
                    level=1,
                    speed="fastest",
                    visuals=False)


    for filename in os.listdir(folder):
        print(filename)
        f = os.path.join(folder, filename)

        print('f', f)
        best = np.loadtxt(f)
        list_gain = list()
        for sim in range(5):
            f,p,e,t = simulation(ENV, best)
            print('FILE: ', filename, 'individual_gain: ', p-e)
            individual_gain = p-e
            list_gain.append(individual_gain)

        gain = np.average(individual_gain)

        list_.append(gain)
  
    tuple_list.append(list_)
print(tuple_list)



fig, ax = plt.subplots()
ax.boxplot(tuple_list)
plt.show()