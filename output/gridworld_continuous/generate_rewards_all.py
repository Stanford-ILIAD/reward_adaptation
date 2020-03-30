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
    if max_ - second_max < 60:
        return second_max_iter
    else:
        return max_iter

dir_name = sys.argv[1]
dir_name = dir_name+'_step_'
print(dir_name)
if 'forward' in dir_name:
    step_num = 16
elif 'backward' in dir_name:
    step_num = 15
required_iter = 0
for i in range(step_num):
    print(dir_name+'{:02d}'.format(i))
    required_iter += list_item(dir_name+'{:02d}'.format(i))
print(required_iter)
