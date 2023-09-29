import sys
from evoman.environment import Environment
from controller import player_controller

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import glob, os

# df1 = pd.read_csv('csv/enemy_1.csv')
# df2 = pd.read_csv('csv/enemy_2.csv')
# df7 = pd.read_csv('csv/enemy_7.csv')


# fig, axs = plt.subplots(1, 3)
# # fig.suptitle('Average from best and mean score per generation over 10 runs')

# axs[0].plot(df1['gen'], df1[['best_avg', 'mean_avg']])
# axs[1].plot(df2['gen'], df2[['best_avg', 'mean_avg']])
# axs[2].plot(df7['gen'], df7[['best_avg', 'mean_avg']])

# axs[0].set_title('Enemy 1')
# axs[1].set_title('Enemy 2')
# axs[2].set_title('Enemy 7')

# axs[0].legend(['Best', 'Mean'])
# axs[1].legend(['Best', 'Mean'])
# axs[2].legend(['Best', 'Mean'])

# for ax in axs.flat:
#     ax.set(xlabel='generations', ylabel='fitness score')

# for ax in axs.flat:
#     ax.label_outer()

# plt.show()


df_GA_1 = pd.read_csv('results_csv/results_GA_1.csv')
df_GA_2 = pd.read_csv('results_csv/results_GA_2.csv')
df_GA_7 = pd.read_csv('results_csv/results_GA_7.csv')
df_DGA_1 = pd.read_csv('results_csv/results_DGA_1.csv')
df_DGA_2 = pd.read_csv('results_csv/results_DGA_2.csv')
df_DGA_7 = pd.read_csv('results_csv/results_DGA_7.csv')



df_GA_1_avg = df_GA_1[['gen', 'best', 'mean']].groupby(['gen']).mean()
df_GA_2_avg = df_GA_2[['gen', 'best', 'mean']].groupby(['gen']).mean()
df_GA_7_avg = df_GA_7[['gen', 'best', 'mean']].groupby(['gen']).mean()
df_DGA_1_avg = df_DGA_1[['gen', 'best', 'mean']].groupby(['gen']).mean()
df_DGA_2_avg = df_DGA_2[['gen', 'best', 'mean']].groupby(['gen']).mean()
df_DGA_7_avg = df_DGA_7[['gen', 'best', 'mean']].groupby(['gen']).mean()

print(df_DGA_2_avg)

fig, axs = plt.subplots(1, 3)
axs[0].plot(df_GA_1_avg.index, df_GA_1_avg[['best', 'mean']])
axs[1].plot(df_GA_2_avg.index, df_GA_2_avg[['best', 'mean']])
axs[2].plot(df_GA_7_avg.index, df_GA_7_avg[['best', 'mean']])

axs[0].set_title('Enemy 1')
axs[1].set_title('Enemy 2')
axs[2].set_title('Enemy 7')

axs[0].legend(['Best', 'Mean'])
axs[1].legend(['Best', 'Mean'])
axs[2].legend(['Best', 'Mean'])

for ax in axs.flat:
    ax.set(xlabel='generations', ylabel='fitness score')

for ax in axs.flat:
    ax.label_outer()
plt.show()



fig, axs = plt.subplots(1, 3)
axs[0].plot(df_DGA_1_avg.index, df_DGA_1_avg[['best', 'mean']])
axs[1].plot(df_DGA_2_avg.index, df_DGA_2_avg[['best', 'mean']])
axs[2].plot(df_DGA_7_avg.index, df_DGA_7_avg[['best', 'mean']])

axs[0].set_title('Enemy 1')
axs[1].set_title('Enemy 2')
axs[2].set_title('Enemy 7')

axs[0].legend(['Best', 'Mean'])
axs[1].legend(['Best', 'Mean'])
axs[2].legend(['Best', 'Mean'])

for ax in axs.flat:
    ax.set(xlabel='generations', ylabel='fitness score')

for ax in axs.flat:
    ax.label_outer()
# plt.show()



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