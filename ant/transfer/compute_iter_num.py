import os
import numpy as np
import sys
dir_ = sys.argv[1]

stages = os.listdir(dir_)
relax_iter = int(dir_.split('/')[-1].split('_')[1].split('rom')[1])
back_iter = 0
for stage in stages:
    files = os.listdir(os.path.join(dir_,stage))
    iter_ids = []
    for file_ in files:
        iter_ids.append(int(file_.split('_')[1]))
    back_iter += np.max(iter_ids)
print((relax_iter+back_iter)*5000)
