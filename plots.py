import sys
from evoman.environment import Environment
from controller import player_controller
from scipy import stats

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import glob, os

enemies = [1,2,7]
experiment = 'GA_optimization'
methods = ['GA', 'DGA']

list_files = [pd.read_csv(experiment + '/results_'+m +'_' + str(e) + '.csv') for e in enemies for m in methods]
[print('{} {}'.format(m, e)) for e in enemies for m in methods]
list_df = []
for df in list_files:
    df_avg = df[['gen', 'best', 'mean', 'diversity']].groupby(['gen']).mean()
    df_avg[['best_std', 'mean_std']] = df[['gen', 'best', 'mean']].groupby(['gen']).std()
    df_avg['lower_best'] = df_avg['best'] - df_avg['best_std']
    df_avg['upper_best'] = df_avg['best'] + df_avg['best_std']
    df_avg['lower_mean'] = df_avg['mean'] - df_avg['mean_std']
    df_avg['upper_mean'] = df_avg['mean'] + df_avg['mean_std']
    list_df.append(df_avg)

# Create line plots
fig, axs = plt.subplots(1, 3)
for df in range(3):
    axs[df].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df].set_title('Enemy {}'.format(enemies[df]))
    axs[df].legend(['Best', 'Mean'], loc='lower right')

for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Gain')

for ax in axs.flat:
    ax.label_outer()
plt.show()

fig, axs = plt.subplots(1, 3)

for df in range(3, 6):
    print(df)
    axs[df-3].plot(list_df[df].index, list_df[df][['best', 'mean']])
    axs[df-3].fill_between(list_df[df].index, list_df[df]['lower_best'], list_df[df]['upper_best'], facecolor='C0', alpha=0.4)
    axs[df-3].fill_between(list_df[df].index, list_df[df]['lower_mean'], list_df[df]['upper_mean'], facecolor='C1', alpha=0.4)
    axs[df-3].set_title('Enemy {}'.format(enemies[df-3]))
    axs[df-3].legend(['Best', 'Mean'], loc='lower right')

for ax in axs.flat:
    ax.set(xlabel='Generation', ylabel='Gain')

for ax in axs.flat:
    ax.label_outer()
plt.show()



filtered_df1 = list_df[4]
filtered_df2 = list_df[5]

# Create a plot for 'div' over 'gen' in the same plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot the data from the first DataFrame
plt.plot(filtered_df1.index, filtered_df1['diversity'], label='GA')

# Plot the data from the second DataFrame
plt.plot(filtered_df2.index, filtered_df2['diversity'], label='DGA')

# Add labels and a legend
plt.xlabel('Generation')
plt.ylabel('Diversity')
plt.title('Diversity over generations for solution {}'.format(1))
plt.legend()

# Show the plot
plt.show()

# Create diversity plots
for i in range(4,7):
    filtered_df1 = list_files[2]
    filtered_df2 = list_files[3]

    filtered_df1 = filtered_df1[filtered_df1['solution'] == i]
    filtered_df2 = filtered_df2[filtered_df2['solution'] == i]
    # Create a plot for 'div' over 'gen' in the same plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Plot the data from the first DataFrame
    plt.plot(filtered_df1['gen'], filtered_df1['diversity'], label='GA')

    # Plot the data from the second DataFrame
    plt.plot(filtered_df2['gen'], filtered_df2['diversity'], label='DGA')

    # Add labels and a legend
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Diversity over generations for solution {}'.format(i))
    plt.legend()

    # Show the plot
    plt.show()

experiment = 'GA_optimization' # name of the experiment
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
                    level=1,
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
for enemy in enemies:
    ENV.update_parameter('enemies', [enemy])
    for method in ['GA', 'DGA']:
        best = np.loadtxt(experiment + '/best_'+method+'_'+str(enemy)+'.txt')
        gains_solutions = []
        for i in range(10):
            gain_sims = []
            cur_best = best[i]
            for sim in range(5):
                f,p,e,t = simulation(ENV, cur_best)
                gain_sims.append(p-e)
            # print('Enemy {} method {} | gain = {}'.format(enemy, method, np.average(gain_sims)))
            gains_solutions.append(np.average(gain_sims))
        gains.append(gains_solutions)
  
import matplotlib.pyplot as plt

for i in range(0, len(enemies)*2, 2):
    t_statistic, p_value = stats.ttest_ind(gains[i], gains[(i+1)])
    print('For enemy {} the p-value is {}'.format(enemies[int(i/2)], p_value))

# Create a boxplot
plt.boxplot(np.array(gains).T)  
plt.title('Gain best GA and DGA solutions vs three enemies')
plt.xlabel('Algorithm and enemy')
plt.ylabel('Gain')
x_labels = ['GA 1', 'DGA 1', 'GA 2', 'DGA 2', 'GA 7', 'DGA 7']
plt.xticks(np.arange(1, 7), x_labels)  

plt.show()