# trial for TS
# Hariya Nobuki

from thompson_sampling import thompson_sampling
from epsilon_greedy import epsilon_greedy
from upper_confidence_bound import UCB
from Arm import Arm
import numpy as np
import pandas as pd

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
    makefiles(os.getcwd())  # cerrent dir

    num_trial = 101
    # complication setteing
    numarms_3 = 5
    numarms_5 = 5
    numarms_7 = 5

    # 形式を変換したら簡単だと思っています
    eg_3_reward = np.array()
    eg_5_reward = np.array()
    eg_7_reward = np.array()
    ucb_reward = np.array()
    ts_reward = np.array()
    for i in range(num_trial):
        if i==0:
            # e-greedy
            eg_3_reward = simulate_eg(0.3)

            eg_5_reward = simulate_eg(0.5)
            eg_7_reward = simulate_eg(0.7)
            # ucb ts
            ucb_reward = simulate_ucb()
            ts_reward = simulate_ts()
    
    print(crayons.red('stats'))
    np_quantile_30 = {m :  np.percentile(results_np[m], 30, axis=0) for m in sample_list}
    np_quantile_70 = {}


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