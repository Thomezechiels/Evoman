import sys
from evoman.environment import Environment
from controller import player_controller
from scipy import stats

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import glob, os

enemy_sets = {
                '1': [1,2,5,7], 
                '2': [3,4,6,8],
            }
experiment = 'generalist_optimization'
methods = ['GI', 'RI']

list_files = [pd.read_csv(experiment + '/results_'+m +'_' + str(set) + '.csv') for m in methods for set, enemies in enemy_sets.items()]
[print('Method {}, set {}'.format(m, set)) for m in methods for set, enemies in enemy_sets.items()]
list_df = []
print(len(list_files))
for df in list_files:
    df_avg = df[['gen', 'best', 'mean']].groupby(['gen']).mean()
    df_avg[['best_std', 'mean_std']] = df[['gen', 'best', 'mean']].groupby(['gen']).std()
    df_avg['lower_best'] = df_avg['best'] - df_avg['best_std']
    df_avg['upper_best'] = df_avg['best'] + df_avg['best_std']
    df_avg['lower_mean'] = df_avg['mean'] - df_avg['mean_std']
    df_avg['upper_mean'] = df_avg['mean'] + df_avg['mean_std']
    list_df.append(df_avg)


# Create line plots
fig, axs = plt.subplots(1, len(enemy_sets))
for df in range(len(enemy_sets)):
    axs[df].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df].set_title('Enemy {}'.format(enemy_sets[str(df + 1)]))
    axs[df].legend(['Best', 'Mean'], loc='lower right')
    axs[df].set_ylim([0, 100])

for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Gain')

for ax in axs.flat:
    ax.label_outer()
plt.show()

fig, axs = plt.subplots(1, 2)

for df in range(2, 4):
    print(df)
    axs[df-2].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df-2].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df-2].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df-2].set_title('Enemy {}'.format(enemy_sets[str(df - 2 + 1)]))
    axs[df-2].legend(['Best', 'Mean'], loc='lower right')
    axs[df-2].set_ylim([0, 100])

for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Gain')

for ax in axs.flat:
    ax.label_outer()
plt.show()

headless = True # True for not using visuals, false otherwise
playermode = "ai"
enemymode = "static"

lb_w, ub_w = -1, 1 # lower and ubber bound weights NN
n_hidden_nodes = 10 # size hidden layer NN
run_mode = 'train' # train or test

ENV = Environment(experiment_name=experiment,
                    playermode=playermode,
                    player_controller=player_controller(n_hidden_nodes),
                    enemymode=enemymode,
                    level=2,
                    multiplemode="yes",
                    speed="fastest",
                    visuals=False)

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment):
    os.makedirs(experiment)
    

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e,t


gains = []
for set, enemies in enemy_sets.items():
    ENV.update_parameter('enemies', enemies)
    for method in methods:
        best = np.loadtxt(experiment + '/best_'+method+'_set_'+str(set)+'.txt')
        gains_solutions = []
        for i in range(10):
            cur_best = best[i]
            f,p,e,t = simulation(ENV, cur_best)
            # print('Enemy {} method {} | gain = {}'.format(enemy, method, np.average(gain_sims)))
            gains_solutions.append(p-e)
        gains.append(gains_solutions)
  
import matplotlib.pyplot as plt

for i in range(0, len(enemy_sets)*2, 2):
    t_statistic, p_value = stats.ttest_ind(gains[i], gains[(i+1)])
    print('For enemy {} the p-value is {}'.format(enemy_sets[str(int((i+2)/2))], p_value))

# Create a boxplot
plt.boxplot(np.array(gains).T)  
plt.title('Gain best GI and RI solutions vs 2 enemy sets')
plt.xlabel('Algorithm and enemy')
plt.ylabel('Gain')
x_labels = ['GI enemy set 1', 'RI enemy set 1', 'GI enemy set 2', 'RI enemy set 2']
plt.xticks(np.arange(1, 5), x_labels)  

plt.show()