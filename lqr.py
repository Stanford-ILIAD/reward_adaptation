"""

LQR local path planning

adapted from: Atsushi Sakai (@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import sklearn
from sklearn import datasets
import scipy

SHOW_ANIMATION = True


class LQRPlanner:

    def __init__(self):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.1  # Time tick
        self.GOAL_DIST = 0.1
        self.MAX_ITER = 150
        self.EPS = 0.01
        self.init_system_models()
        self.time = 0.0

    def init_system_models(self):
        """
        Initialize linear system models
        """
        self.A = np.array([[self.DT, 1.0],
                           [0.0, self.DT]])
        self.B = np.array([0.0, 1.0]).reshape(2, 1)
        self.R = np.eye(1)
        Q1 = np.array([[ 2.33035357, -0.81803898], [-0.81803898,  1.17972475]])
        #self.Q = sklearn.datasets.make_spd_matrix(2, random_state=None)
        self.Q = Q1

    def lqr_planning(self, sx, sy, gx, gy, Q, show_animation=True):

        rx, ry = [sx], [sy]

        x = np.array([sx - gx, sy - gy]).reshape(2, 1)  # State vector

        found_path = False
        self.Q = Q
        print("\nQ: ", self.Q)

        self.time = 0.0
        while self.time <= self.MAX_TIME:
            self.time += self.DT
            #print("--Time: ", self.time)

            u = self.lqr_control(x)

            x = self.A @ x + self.B @ u

            rx.append(x[0, 0] + gx)
            ry.append(x[1, 0] + gy)
            #if time == self.DT:
            #    print("Ax: ", self.A@x)
            #    print("Bu: ", self.B@u)
            #    print("x_t+1: ", x)
            #    print("posx, posy: ", x[0,0] + gx, x[1,0] + gy)

            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            if d <= self.GOAL_DIST:
                found_path = True
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            print("Cannot found path")
            return [], []

        return rx, ry

    def solve_dare(self):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        X, Xn = self.Q, self.Q

        for i in range(self.MAX_ITER):
            Xn = self.A.T @ X @ self.A - self.A.T @ X * self.B * \
                 la.inv(self.R + self.B.T @ X @ self.B) * self.B.T @ X @ self.A + self.Q
            print("\nAA: ", self.A.T@self.A)
            print("APA: ", self.A.T@X@self.A)
            print("BPA: ", self.B.T@X@self.A)
            print("inv: ", la.inv(self.R + self.B.T @ X @ self.B) )
            print("inv*BA: ", la.inv(self.R + self.B.T @ X @ self.B) * self.B.T @ X @ self.A )
            print("AB*inv*BA: ", self.A.T @ X * self.B * \
                 la.inv(self.R + self.B.T @ X @ self.B) * self.B.T @ X @ self.A )
            print("evthing cept Q: ", self.A.T @ X @ self.A - self.A.T @ X * self.B * \
                 la.inv(self.R + self.B.T @ X @ self.B) * self.B.T @ X @ self.A)
            print("Pt-1: ", Xn, i)
            if (abs(Xn - X)).max() < self.EPS:
                break
            X = Xn

        return Xn

    def dlqr(self):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        #X1 = self.solve_dare()
        #print("X1: ", X1)
        X = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        if self.time == self.DT: print("X: ", X)

        # compute the LQR gain
        #K = la.inv(self.B.T @ X @ self.B + self.R) @ (self.B.T @ X @ self.A)
        K = np.matrix(scipy.linalg.inv(self.B.T * X * self.B + self.R) * (self.B.T * X * self.A))
        #print("BXA: ", self.B.T * X * self.A)
        #print("BXB: ",self.B.T * X * self.B)
        #print("inv: ", scipy.linalg.inv(self.B.T * X * self.B + self.R))
        #print("inv*BXA: ", scipy.linalg.inv(self.B.T * X * self.B + self.R) * (self.B.T * X * self.A))
        if self.time == self.DT: print("K: ", K)
        eigValues = la.eigvals(self.A - self.B @ K)

        return K, X, eigValues



    def lqr_control(self, x):

        Kopt, X, ev = self.dlqr()

        u = -Kopt @ x  # normal
        #print("u: ", u)

        return u


def main():
    print(__file__ + " start!!")

    ntest = 10  # number of goal
    #area = 100.0  # sampling area

    lqr_planner = LQRPlanner()
    Qs = [np.eye(2)*0.5, np.eye(2), np.eye(2)*2, np.eye(2)*3, np.eye(2)*4, np.eye(2)*5]
    #Qs = [np.array([[ 1. , -0.9],
    #                [-0.9,  1. ]]),
    #    np.array([[ 1. , -0.5],
    #            [-0.5,  1. ]]),
    #    np.array([[1. , 0.5],
    #            [0.5, 1. ]]),
    #    np.array([[1. , 0.9],
    #            [0.9, 1. ]])]

    for i in range(len(Qs)):
        sx = 0.0
        sy = 0.0
        #gx = random.uniform(-area, area)
        #gy = random.uniform(-area, area)
        gx = 50.
        gy = 50.

        rx, ry = lqr_planner.lqr_planning(sx, sy, gx, gy, Qs[i], show_animation=SHOW_ANIMATION)

        if SHOW_ANIMATION:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)


if __name__ == '__main__':
    main()

