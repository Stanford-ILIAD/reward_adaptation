import os
import sys
import numpy as np

def sort_fun(elems):
    return elems[0]

def list_item(dir_name):
    names = os.listdir(dir_name)
    inames = [[int(item.split('_')[1]), float(item.split('_')[2].split('.')[0])] for item in names if ('best_model' not in item) and ('final' not in item) and ('csv' not in item)]
    inames.sort(key=sort_fun)
    max_ = -np.inf
    second_max = max_
    max_iter = 0
    second_max_iter = 0 
    for item in inames:
        if item[1] > max_:
            print('{} {}'.format(item[0],item[1]))
            second_max = max_
            second_max_iter = max_iter
            max_ = item[1]
            max_iter = item[0]
    for item in inames:
        if item[1] > max_-100 or item[1]>3000.:
            break
    return item[0]

dir_name = sys.argv[1]
dir_name = dir_name+'_step_'
print(dir_name)
if 'forward' in dir_name or 'backward' in dir_name or 'barrier_set_size' in dir_name:
    if os.path.exists(dir_name+'_step_15'):
        step_num = 16
    else:
        step_num = 15
elif 'reward_w' in dir_name:
    step_num = 7
required_iter = 0
for i in range(step_num):
    print(dir_name+'{:02d}'.format(i))
    required_iter_  = list_item(dir_name+'{:02d}'.format(i))
    print('iter: ', required_iter_)
    required_iter += required_iter_
print('total number of iterations', required_iter)
