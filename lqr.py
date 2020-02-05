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

SHOW_ANIMATION = True


class LQRPlanner:

    def __init__(self):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.1  # Time tick
        self.GOAL_DIST = 0.1
        self.MAX_ITER = 150
        self.EPS = 0.01
        self.init_system_models()

    def init_system_models(self):
        """
        Initialize linear system models
        """
        self.A = np.array([[self.DT, 1.0],
                           [0.0, self.DT]])
        self.B = np.array([0.0, 1.0]).reshape(2, 1)
        self.R = np.eye(1)
        Q1 = np.array([[ 2.33035357, -0.81803898], [-0.81803898,  1.17972475]])
        Q2 = np.eye(2)
        #self.Q = sklearn.datasets.make_spd_matrix(2, random_state=None)
        self.Q = Q1
        print("Q: ", self.Q)

    def lqr_planning(self, sx, sy, gx, gy, show_animation=True):

        rx, ry = [sx], [sy]

        x = np.array([sx - gx, sy - gy]).reshape(2, 1)  # State vector

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            u = self.lqr_control(x)

            x = self.A @ x + self.B @ u

            rx.append(x[0, 0] + gx)
            ry.append(x[1, 0] + gy)

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
            Xn = self.A.T * X * self.A - self.A.T * X * self.B * \
                 la.inv(self.R + self.B.T * X * self.B) * self.B.T * X * self.A + self.Q
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
        X = self.solve_dare()

        # compute the LQR gain
        K = la.inv(self.B.T @ X @ self.B + self.R) @ (self.B.T @ X @ self.A)

        eigValues = la.eigvals(self.A - self.B @ K)

        return K, X, eigValues



    def lqr_control(self, x):

        Kopt, X, ev = self.dlqr()

        u = -Kopt @ x

        return u


def main():
    print(__file__ + " start!!")

    #ntest = 10  # number of goal
    #area = 100.0  # sampling area

    lqr_planner = LQRPlanner()

    #for i in range(ntest):
    sx = 0.0
    sy = 0.0
    #gx = random.uniform(-area, area)
    #gy = random.uniform(-area, area)
    gx = 50.
    gy = 50.

    rx, ry = lqr_planner.lqr_planning(sx, sy, gx, gy, show_animation=SHOW_ANIMATION)

    if SHOW_ANIMATION:  # pragma: no cover
        plt.plot(sx, sy, "or")
        plt.plot(gx, gy, "ob")
        plt.plot(rx, ry, "-r")
        plt.axis("equal")
        plt.pause(1.0)


if __name__ == '__main__':
    main()

