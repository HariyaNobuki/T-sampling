# trial for TS
# Hariya Nobuki

from thompson_sampling import thompson_sampling
from epsilon_greedy import epsilon_greedy
from upper_confidence_bound import UCB
from Arm import Arm
import numpy as np

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

    num_trial = 100
    # complication setteing
    numarms_3 = 5
    numarms_5 = 5
    numarms_7 = 5

    # 統計も同時でいいような気がしています
    eg_3_reward_hist = []
    eg_5_reward_hist = []
    eg_7_reward_hist = []
    ucb_reward_hist = []
    ts_reward_hist = []
    for i in range(num_trial):
        # e-greedy
        eg_3_reward_hist.append(simulate_eg(0.3))
        eg_5_reward_hist.append(simulate_eg(0.5))
        eg_7_reward_hist.append(simulate_eg(0.7))
        # ucb ts
        ucb_reward_hist.append(simulate_ucb())
        ts_reward_hist.append(simulate_ts())

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