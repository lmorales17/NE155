# NE 155: HW 4, Problem 4
#
#
# @author: Luis Morales

import scipy as sp
import numpy as np
from scipy import sparse
from scipy import linalg
from matplotlib.pyplot import *

### Problem 4 ###

# a) Use built in Python or MATLAB commands to construct A and b.



#Using the values given in the problem

n = 100

values = [-1.0, 2.0, -1.0]

diagonals = [-1.0, 0.0, 1.0]

A_matrix = sp.sparse.diags(values, diagonals, shape = (n, n)).todense()
b_vector = np.array(xrange(1, n + 1))

print "\nThis is the A matrix produced using built-in python packages:\n{0}\n".format(A_matrix)

print "\nThis is the b vector produced using built-in python packages:\n{0}\n".format(b_vector)

# (b) What is the condition number of A?
print "\nCondition Number of A is equal to {:.6}\n".format(np.linalg.cond(A_matrix))

# (c) Solve by explicit inversion

x_inversion = np.dot(sp.linalg.inv(A_matrix), b_vector)
print "\nSolution by explicit inversion:\n{0}\n".format(x_inversion)

# (d) Solve with linalg.solve
x_solve = sp.linalg.solve(A_matrix, b_vector)
print "Solution with the built-in \'linalg.solve\' function:\n{0}\n".format(x_solve)

# Now here are the plots
plot(x_inversion, 'r')
plot(x_solve, 'o')
legend(("inverse", "solver"))
title("X Vector Solutions of Matrix A using both \'linalg.inv\' and \'linalg.solve\'")
xlabel("X Values")
ylabel("Matrix Solution Values")
show()

