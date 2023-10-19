import sys
from evoman.environment import Environment
from controller import player_controller
from scipy import stats

import types

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

    return vfitness, vplayerlife, venemylife, vtime
 
ENV.fitness_single = types.MethodType(fitness_single, ENV)
ENV.multiple = types.MethodType(multiple, ENV)   

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e,t


fig = plt.figure()
ax = fig.add_subplot(111)

all_enemies = [1,2,3,4,5,6,7,8]
gains = []
ENV.update_parameter('enemies', all_enemies)
set='1'
for method in methods:
    best = np.loadtxt(experiment + '/best_'+method+'_set_'+str(set)+'.txt')
    gains_solutions = []
    for i in range(len(best)):
        cur_best = best[i]
        f,p,e,t = simulation(ENV, cur_best)
        # print('Enemy {} method {} | gain = {}'.format(enemy, method, np.average(gain_sims)))
        gains_solutions.append(sum(p)-sum(e))
    print('Higest gain method {} enemy set {}: {:.2f} from solution {}'.format(method, str(set), np.max(gains_solutions), str(np.argmax(gains_solutions)+1)))
    gains.append(gains_solutions)
bp = ax.boxplot(np.array(gains).T, positions=[1,2], widths = 0.4, patch_artist=True, boxprops=dict(facecolor="lightblue"), zorder=1)

for i, d in enumerate(np.array(gains).T):
   for j in range(len(d)):
       y = d[j]
       ax.scatter(j+1, y, color='gray', edgecolors='black', alpha = 0.4, zorder=2)

set='2'
gains = []
for method in methods:
    best = np.loadtxt(experiment + '/best_'+method+'_set_'+str(set)+'.txt')
    gains_solutions = []
    for i in range(len(best)):
        cur_best = best[i]
        f,p,e,t = simulation(ENV, cur_best)
        # print('Enemy {} method {} | gain = {}'.format(enemy, method, np.average(gain_sims)))
        gains_solutions.append(sum(p)-sum(e))
    print('Higest gain method {} enemy set {}: {:.2f} from solution {}'.format(method, str(set), np.max(gains_solutions), str(np.argmax(gains_solutions)+1)))
    gains.append(gains_solutions)
bp1 = ax.boxplot(np.array(gains).T, positions=[3,4], widths = 0.4, patch_artist=True, boxprops=dict(facecolor="steelblue"), zorder=1)

for i, d in enumerate(np.array(gains).T):
   for j in range(len(d)):
       y = d[j]
       ax.scatter(j+3, y, color='gray', edgecolors='black', alpha = 0.4, zorder=2)


ax.legend([bp["boxes"][0], bp1["boxes"][0]], [enemy_sets['1'], enemy_sets['2']], loc='upper right')


# Create a boxplot
# plt.boxplot(np.array(gains).T)  
plt.title('Gain best GI and RI solutions')
plt.xlabel('Evolutionary algorithm')
plt.ylabel('Gain')
x_labels = ['GI', 'RI', 'GI', 'RI']
plt.xticks(np.arange(1, 5), x_labels)  

plt.show()

# Very best solution (player life and enemy life per enemy)  
very_best = np.loadtxt(experiment + '/best_GI_set_1.txt')[1]
f,p,e,t = simulation(ENV, very_best)
print('Player life', p)
print('Enemy life', e)

# # t-test for significance 
# for i in range(0, len(enemy_sets)*2, 2):
#     t_statistic, p_value = stats.ttest_ind(gains[i], gains[(i+1)])
#     print('For enemy {} the p-value is {}'.format(enemy_sets[str(int((i+2)/2))], p_value))