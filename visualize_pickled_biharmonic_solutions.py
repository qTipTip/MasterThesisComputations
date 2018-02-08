"""
Visualizes all (or select ones) pickled solutions in the
biharmonic_pickled_solutions folder.
"""

import pickle
import PSFEM
import SSplines

import numpy as np
import matplotlib.pyplot as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from biharmonic import read_pickled_solutions
from itertools import product

solutions = read_pickled_solutions()

res = 100
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
X, Y = np.meshgrid(x, y)

def plot_solution(solution, zlim=3, savefig=True, display=False):

    # unpack solution
    n, M, V, u_approx = solution

    fig = plt.figure()
    axs = Axes3D(fig)
    axs.set_zlim3d(-zlim, zlim)
    plt.title('FEM-approximation using {:d} elements'.format(len(M.triangles)))

    # evaluate approximate solution
    z = np.zeros((res, res))

    for i, xp in enumerate(x):
        for j, yp in enumerate(y):
            p = np.array([xp, yp])
            k = M.find_triangle(p)
            z[j, i] = u_approx(p, k)

    # plot solution
    axs.plot_surface(X, Y, z)

    if savefig:
        filename = 'pictures/biharmonic_approximation_values/biharmonic_approximation_{:04d}.pdf'.format(n)
        plt.savefig(filename)
    if display:
        plt.show()

if len(sys.argv) > 1:
    for i in sys.argv[1:]:
        solution = solutions[int(i)-2]
        print(solution)
        plot_solution(solution, savefig=True, display=False)
else:
    for solution in solutions:
        plot_solution(solution, savefig=True, display=False)
