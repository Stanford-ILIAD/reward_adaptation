import numpy as np
import os
import eval_model as eval
import scipy

h2v0 = ("gridworld", "h2_v0", "best_model_5000_11.62.pkl")
h2v1 = ("gridworld", "h2_v1", "best_model_88000_11.559999999999999.pkl")
h1v0 = ("gridworld", "h1_v0", "best_model_6000_11.62.pkl")
h1v1 = ("gridworld", "h1_v1", "best_model_17000_11.52.pkl")

def eigenvector_sim(m1, m2):
    param_names = [
        "deepq/model/action_value/fully_connected_2/weights:0",
                "deepq/model/action_value/fully_connected_1/weights:0",
                    "deepq/model/action_value/fully_connected/weights:0",
    ]
    for param_name in param_names:
        W1 = m1.get_parameters()[param_name]
        W2 = m2.get_parameters()[param_name]
        U1, S1, V1 = np.linalg.svd(W1)
        U2, S2, V2 = np.linalg.svd(W2)
        cos_dists = []
        print(U1.shape[1])
        for i in range(U1.shape[1]):
            #print("Evec : ", i)
            #print(U1, U1[:, i])
            #print(U2, U2[:,i])
            cos_dist = scipy.spatial.distance.cosine(U1[:,i], U2[:,i])
            #print("Evec: ", i, ", cos dist: ", cos_dist )
            cos_dists.append(cos_dist)
        print("mean cos dists: ", np.mean(cos_dists))
        #print("S1: ", S1)
        break


def qvalue_sim(m1, m2):
    dists = []
    h2v0 = [[0,0], [0,1], [0,2], [0,3], [0,4], [1,4], [2,4], [3,4], [4,4]]
    for i in range(5):
        for j in range(5):
            S = np.array([i,j])
            S = S.reshape([1,2])
            print("\nstate: ", S[0])
            actions1, qvalues1, _ = m1.step_model.step(obs=S)
            actions2, qvalues2, _ = m2.step_model.step(obs=S)
            actionp1 = m1.action_probability(S)
            actionp2 = m2.action_probability(S)

            #l1_dist = np.linalg.norm(qvalues1[0]-qvalues2[0])
            #l1_dist = np.abs(qvalues1[0][actions1[0]] - qvalues2[0][actions1[0]])
            #state_dist = np.abs(max(qvalues1[0]) - max(qvalues2[0]))
            state_dist = np.abs(np.dot(actionp1[0], qvalues1[0]) - np.dot(actionp2[0], qvalues2[0]))
            #kl = scipy.special.kl_div(qvalues1[0], qvalues2[0])
            #print("l1 dist: ", l1_dist)
            #dists.append(l1_dist)
            #if actions1[0]==actions2[0]:
            #    dists.append(1)

            #if [i,j] in h2v0:
            print("actions: ", actions1[0], actions2[0])
            print("qvalues: ", qvalues1[0], qvalues2[0])
            print("dist: ", state_dist)
            dists.append(state_dist)
    print("mean dists: ", np.mean(dists))


def main():
    for m1_info in [h2v0, h2v1, h1v0, h1v1]:
        for m2_info in [h2v0, h2v1, h1v0, h1v1]:
            if m1_info != m2_info:
                print()
                m1_info = h1v1
                m2_info = h2v0
                print(m1_info[1], m2_info[1])
                m1_dir = os.path.join(m1_info[0], m1_info[1], m1_info[2])
                m1 = eval.load_model(m1_dir)
                m2_dir = os.path.join(m2_info[0], m2_info[1], m2_info[2])
                m2 = eval.load_model(m2_dir)
                #eigenvector_sim(m1, m2)
                qvalue_sim(m1, m2)
                break
        break

if __name__=='__main__':
    main()
