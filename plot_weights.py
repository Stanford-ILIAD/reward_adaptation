import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from stable_baselines import PPO2
from utils import *
# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

RS = 20150101
NGROUPS = 3

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(NGROUPS):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.show()
    return f, ax, sc, txts


def extract_weights(group):
    weights = []
    for info in group:
        model_dir = os.path.join(info[0], info[1], info[2])
        model = PPO2.load(model_dir)
        #for name in model.get_parameter_list():
        #    print(name)
        weight = model.get_parameters()['model/vf/w:0']
        weights.append(weight)
    return weights


def main():
    group1 = [weight_n1, weight_n05]
    group2 = [weight_0, weight_p05, weight_p1]
    group3 = [weight_p2, weight_p4, weight_p6, weight_p8, weight_p10, weight_p100]
    weight_group1 = extract_weights(group1)
    weight_group2 = extract_weights(group2)
    weight_group3 = extract_weights(group3)
    X = []
    weight_labels = []
    for i,wg in enumerate([weight_group1, weight_group2, weight_group3]):
        X.extend(wg)
        weight_labels.append([i]*len(wg))
    X = np.concatenate(X)
    weight_labels = np.concatenate(weight_labels)
    X = X.reshape((11, 64*1))
    weights_proj = TSNE(random_state=RS).fit_transform(X)
    scatter(weights_proj, weight_labels)

if __name__=='__main__':
    main()