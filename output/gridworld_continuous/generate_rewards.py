import os
import sys
import numpy as np

def sort_fun(elems):
    return elems[0]

dir_name = sys.argv[1]
names = os.listdir(dir_name)
inames = [[int(item.split('_')[1]), float(item.split('_')[2].split('.')[0])] for item in names if ('best_model' not in item) and ('final' not in item) and ('csv' not in item)]
inames.sort(key=sort_fun)
max_ = -np.inf
for item in inames:
    if item[1] > max_:
        print('{} {}'.format(item[0],item[1]))
        max_ = item[1]
