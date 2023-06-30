import numpy as np
import matplotlib.pyplot as plt
import os

def plot_result(trials_avg_rewards, max_steps, eval_freq, env_name, save_path):
    trials_avg_rewards = np.array(trials_avg_rewards)
    std = np.std(trials_avg_rewards, axis=0)
    mean = np.mean(trials_avg_rewards, axis=0)
    x = [i * eval_freq for i in range(max_steps // eval_freq + 1)]

    plt.title(env_name)
    plt.xlabel('Time steps')
    plt.ylabel('Average Return')
    plt.plot(x, mean, color='#2277aa')
    plt.fill_between(x, mean + std, mean - std, color = '#a8d1df', alpha=0.4)

    plt.savefig(os.path.join(save_path, 'plot.jpg')) 