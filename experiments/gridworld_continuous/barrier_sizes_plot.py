"""
Plots distance of fine-tuned trajectory to target trajectory over training iterations.
We compare different barrier sizes with the hypothesis that smaller barrier sizes
should result in smaller distances.
"""
import pandas as pd
from eval_model import *
from utils import *
from output.gridworld_continuous.policies import *

barrier_sizes = [(barrier02_L1, barrier02_R1),
                 (barrier1_L1, barrier1_R1),
                 (barrier2_L1, barrier2_R1),
                 (barrier3_L1, barrier3_R1),
                 (barrier6_L1, barrier6_R1)]


def l1_dist(traj1, traj2):
    if len(traj1) < len(traj2):
        longer_traj, shorter_traj = traj2, traj1
    else:
        longer_traj, shorter_traj = traj1, traj2
    return np.linalg.norm(longer_traj[:len(shorter_traj)] - shorter_traj, 1)


def get_single_traj(model_info):
    eval_env = load_env("Continuous-v0", "PPO")
    model_dir = os.path.join(model_info[0], model_info[1], model_info[2])
    model = load_model(model_dir)
    _, _, _, state_history = evaluate(model, eval_env, render=False)
    return state_history


def main():
    # initialize csv file
    path = "output/gridworld_continuous/barrier_sizes_plot.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["barrier_size", "training_step", "norm_dist"])

    for (Linfo, Rinfo) in barrier_sizes:

        # get trajectories for L, R
        Lstate_history = get_single_traj(Linfo)
        Rstate_history = get_single_traj(Rinfo)

        # read fine-tuned trajectories
        cols = []
        for col_no in range(1, 100):
            cols.append(excel_style(col_no))
        FTmodel = Rinfo[1] + "_L1"
        df = pd.read_csv("output/gridworld_continuous/{}/trajs.csv".format(FTmodel), names=cols)

        training_steps = []
        trajs = []
        for ridx, row in df.iterrows():
            traj = []
            for cidx, col in enumerate(list(df)):
                elem = df.iloc[ridx, cidx]
                if cidx == 0:
                    training_steps.append(elem)
                else:
                    if pd.isna(elem):  # if we find Nan, just repeat last element
                        traj.append(traj[-1])
                    else:
                        elem = np.array(elem[1:-1].split()).astype(float)
                        traj.append(elem)
            trajs.append(np.stack(traj))
        print(len(training_steps), len(trajs))

        # metric function: L1 distance between trajectories
        for i in range(len(trajs)):
            training_step = training_steps[i]
            traj = trajs[i]
            # normalize
            max_dist = l1_dist(Lstate_history, Rstate_history)
            norm_dist = l1_dist(traj, Lstate_history) / max_dist
            #norm_dist = l1_dist(traj, Lstate_history)

            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([Linfo[1][:8], training_step, norm_dist])
        # for each training step, plot distance (metric function)
        #break


if __name__ == "__main__":
    main()
