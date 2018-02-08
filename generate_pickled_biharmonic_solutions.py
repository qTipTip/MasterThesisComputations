import pickle
import biharmonic
import PSFEM

import os
import sys
import shutil

"""
Computes solutions to the biharmonic equation and pickles them.
"""

n_values = map(int, sys.argv[1:])

for n in n_values:
    filename = 'biharmonic_pickles/biharmonic_solution_{:04d}.p'.format(n)

    if os.path.isfile(filename):
        ans = input(filename + ' already exists. Overwrite? Y/N')
        if ans == 'N':
            continue

    M = PSFEM.unit_square_uniform(n)
    V = PSFEM.CompositeSplineSpace(M)
    u_approx = PSFEM.solve(biharmonic.a, biharmonic.L, V, verbose=True)
    solution_data = (n, M, V, u_approx)

    print('Saving solution to file.')
    pickle.dump(solution_data, open(filename, 'wb'))
