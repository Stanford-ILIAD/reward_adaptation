from eval_model import *
from gridworld_policies.policies import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt2


#fdir = "output/gridworld/perturbations"
#fname = "eval.csv"
#fpath = os.path.join(fdir, fname)
#if os.path.exists(fpath):
#  os.remove(fpath)
#with open(fpath, 'a') as f:
#  writer = csv.writer(f)
#  writer.writerow(["model", "x", "y"])

for model_info in [h2v0, h2v1, h1v0, h1v1]:
    model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
    eval_env = load_env("Gridworld-v0")
    model = load_model(model_dir)
    max_dist = 8.0

    #param_name = "deepq/target_q_func/model/action_value/fully_connected_2/weights:0"  # doesn't change policy
    #param_name = "deepq/target_q_func/model/state_value/fully_connected_2/weights:0"   # doesn't change policy
    #param_name = "deepq/model/state_value/fully_connected_2/weights:0"                 # doesn't change policy
    param_name = "deepq/model/action_value/fully_connected_2/weights:0"                # CHANGES POLICY!

    print("\nrunning original model")
    _, _, _, original_state_history = evaluate(model, eval_env)

    # perturb model
    x = []
    y = []
    for std in np.arange(0, 1.0, 0.1):
         print("\nrunning perturbed model with std {}".format(std))
         perturbed_model = load_model(model_dir)
         weight_before = perturbed_model.get_parameters()[param_name]
         param_idx = get_param_idx(perturbed_model, param_name)
         perturbed_model.sess.run(add_random_noise(perturbed_model.params[param_idx], stddev=std))
         weight_after = perturbed_model.get_parameters()[param_name]
         print("l1 dist sum: ", np.sum(np.abs(weight_before-weight_after)))
         #assert (weight_before != weight_after).all()
         _, _, _, state_history = evaluate(perturbed_model, eval_env)
         added_state_history = np.array([state_history[-1] for _ in range(len(original_state_history) - len(state_history))])
         if len(added_state_history) > 0:
             state_history = np.concatenate((state_history, added_state_history))

         state_dist = np.linalg.norm(original_state_history-state_history, 1)
         print("state_dist: ", state_dist)
         x.append(std)
         y.append(state_dist)

#    with open(fpath, 'a') as f:
#        writer = csv.writer(f)
#        for row in range(len(x)):
#            writer.writerow([model_info[1], x[row], y[row]])
    break

