# trial for TS
# Hariya Nobuki

from thompson_sampling import thompson_sampling
from epsilon_greedy import epsilon_greedy
from upper_confidence_bound import UCB
from Arm import Arm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os , sys
import crayons

def simulate_eg(epsilon):
    arms = []
    for i in range(numarms_3):
        arms.append(Arm(0.3))
    for i in range(numarms_5):
        arms.append(Arm(0.5))
    for i in range(numarms_7):
        arms.append(Arm(0.7))
    return epsilon_greedy(arms=arms, T=10**3, epsilon=epsilon)

def simulate_ucb():
    arms = []
    for i in range(numarms_3):
        arms.append(Arm(0.3))
    for i in range(numarms_5):
        arms.append(Arm(0.5))
    for i in range(numarms_7):
        arms.append(Arm(0.7))
    return UCB(arms=arms, T=10**3)

def simulate_ts():
    arms = []
    for i in range(numarms_3):
        arms.append(Arm(0.3))
    for i in range(numarms_5):
        arms.append(Arm(0.5))
    for i in range(numarms_7):
        arms.append(Arm(0.7))
    return thompson_sampling(arms=arms, T=10**3)

def __out_benchmark(array):
    print('\t Avg: {}, Std: {}, Max: {}, Min: {}'.format(np.average(array), np.std(array), np.max(array), np.min(array)))

def makefiles(path):
    print(crayons.red('make files'))
    for sam in sample_list:
        os.makedirs(path+"/"+sam,exist_ok=True)



if __name__ == "__main__":
    sample_list = ['eg_3','eg_5','eg_7','ucb','ts']
    draw_type == 'ERRORBAR'
    makefiles(os.getcwd())  # cerrent dir

    num_trial = 11
    # complication setteing
    numarms_3 = 5
    numarms_5 = 5
    numarms_7 = 5

    # 形式を変換したら簡単だと思っています
    eg_3_reward = np.array([])
    eg_5_reward = np.array([])
    eg_7_reward = np.array([])
    ucb_reward = np.array([])
    ts_reward = np.array([])
    for i in range(num_trial):
        if i==0:
            # e-greedy
            eg_3_reward = np.array([simulate_eg(0.3)])
            eg_5_reward = np.array([simulate_eg(0.5)])
            eg_7_reward = np.array([simulate_eg(0.7)])
            # ucb ts
            ucb_reward = np.array([simulate_ucb()])
            ts_reward = np.array([simulate_ts()])
        else:
            # e-greedy
            eg_3_reward = np.append(eg_3_reward,[simulate_eg(0.3)],axis=0)
            eg_5_reward = np.append(eg_5_reward,[simulate_eg(0.5)],axis=0)
            eg_7_reward = np.append(eg_7_reward,[simulate_eg(0.7)],axis=0)
            # ucb ts
            ucb_reward = np.append(ucb_reward,[simulate_ucb()],axis=0)
            ts_reward = np.append(ts_reward,[simulate_ts()],axis=0)
    # sample list
    results_np = {'eg_3':eg_3_reward,'eg_5':eg_5_reward,'eg_7':eg_7_reward,'ucb':ucb_reward,'ts':ts_reward}
    print(crayons.red('stats'))
    np_mean = {m :  np.percentile(results_np[m], 0.5, axis=0) for m in sample_list}
    np_quantile_30 = {m :  np.percentile(results_np[m], 0.25, axis=0) for m in sample_list}
    np_quantile_70 = {m :  np.percentile(results_np[m], 0.75, axis=0) for m in sample_list}

    fig, ax = plt.subplots(1)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    # ax.set_yticks(np.arange(92.5, 94.4, 0.2))
    ax.set_yticks(np.arange(93.1, 94.2, 0.2))
    ax.grid(True)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    plt.grid(b=True, which='major', color='#666699', linestyle='--')
    fig.savefig(os.path.join(root_path, 'Accuracy.png'))

    # data science

    #print('Epsilon-greedy_0.3')
    #__out_benchmark(eg_3_reward_hist)
    #print('Epsilon-greedy_0.5')
    #__out_benchmark(eg_5_reward_hist)
    #print('Epsilon-greedy_0.7')
    #__out_benchmark(eg_7_reward_hist)
    #print('UCB')
    #__out_benchmark(ucb_reward_hist)
    #print('Thompson sampling')
    #__out_benchmark(ts_reward_hist)