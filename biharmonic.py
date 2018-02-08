"""
Contains the linear, and bilinear form for the biharmonic equation, as well as
the exact source term and solution to the model problem. Also contains a method
for reading pickled solutions to the biharmonic problem.
"""

import numpy as np

def u_exact(p):
    """
    The exact solution to the model problem for the biharmonic equation.
    """
    x, y = p

    return 4*np.sin(np.pi*x)**2*np.sin(np.pi*y)**2*np.sin(2*np.pi*(y-x))
def f(p):
    """
    The exact source term for the model problem for the biharmonic equation.
    """
    x, y = p
    return 8*np.pi**4 *( np.sin(2*np.pi*x) - 8*np.sin(4*np.pi*x) + 25*np.sin(2*np.pi*(x - 2*y)) -
                     32*np.sin(4*np.pi*(x - y)) - np.sin(2*np.pi*y) + 8*(np.sin(4*np.pi*y) +
                             np.sin(2*np.pi*(-x + y))) + 25*np.sin(4*np.pi*x - 2*np.pi*y))

def a(u, v):
    """
    Given two basis splines on the PS12, computes the L2 inner product of their gradients.
    """
    def lhs(p):
        return u.lapl(p) * v.lapl(p)
    return lhs

def L(v):
    """
    Given a source term f, computes the L2 inner product against a basis spline
    on the PS12.
    """
    def rhs(p):
        return v(p)*f(p)
    return rhs



def read_pickled_solutions():
    """
    Reads pickled solutions to the biharmonic equation and returns them as a
    list.
    """
    import pickle
    import os

    solutions = []
    cwd = os.getcwd() + '/biharmonic_pickles'
    for file in os.listdir('biharmonic_pickles'):
        file_obj = open(cwd + '/' + file, 'rb')
        solutions.append(pickle.load(file_obj))

    return solutions
