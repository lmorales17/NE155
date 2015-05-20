# NE 155: HW 5, Problem 4
#
#
# @author: Luis Morales

import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg
import sys as sys
from matplotlib.pyplot import *

def generateA(values, diagonals):
    return sp.sparse.diags(values, diagonals, shape = (5, 5)).todense()

b = 100.0 * np.ones(5)

def Jacobi(A, b, tolerance = 0.0000001, n = 5):

  x_k = np.zeros(n)
  x_prev = np.zeros(n)

  err = sys.maxint

  num_iterations = 0

  for _ in range(1, 10001):
      if (err < tolerance):
          num_iterations = _
          break;

      for i in range(0, n):
          first_sum = 0.0
          secondary_sum = 0.0

          for j in range(i):
              first_sum = first_sum + A[i, j] * x_prev[j]

          for j in range(i + 1, n):
              secondary_sum = secondary_sum + A[i, j] * x_prev[j]

          x_k[i] = 1.0 / A[i, i] * (b[i] - first_sum - secondary_sum)
      err = np.linalg.norm(np.abs(x_k - x_prev),2) / np.linalg.norm(x_k, 2)
      x_prev = np.array(x_k)

  return (x_k, num_iterations, err)

def Gauss_Seidel(A, b, tolerance = 0.0000001, n = 5):

  x_k = np.zeros(n)
  x_prev = np.zeros(n)

  err = sys.maxint

  num_iterations = 0
  
  for _ in range(1, 10001):
      if (err < tolerance):
          num_iterations = _
          break;

      for i in range(0, n):
          first_sum = 0.0
          secondary_sum = 0.0

          for j in range(i):
              first_sum = first_sum + A[i, j] * x_k[j]

          for j in range(i + 1, n):
              secondary_sum = secondary_sum + A[i, j] * x_prev[j]

          x_k[i] = 1.0 / A[i, i] * (b[i] - first_sum - secondary_sum)
      err = np.linalg.norm(np.abs(x_k - x_prev),2) / np.linalg.norm(x_k, 2)
      x_prev = np.array(x_k)

  return (x_k, num_iterations, err)

def SOR(A, b, omega = 1.1, tolerance = 0.0000001, n = 5):

  x_k = np.zeros(n)
  x_prev = np.zeros(n)
  err = sys.maxint
  
  num_iterations = 0

  for _ in range(1, 10001):
      if (err < tolerance):
          num_iterations = _
          break;

      for i in range(0, n):
          first_sum = 0.0
          secondary_sum = 0.0

          for j in range(i):
              first_sum = first_sum + A[i,j] * x_k[j]

          for j in range(i + 1, n):
              secondary_sum = secondary_sum + A[i,j] * x_prev[j]

          x_k[i] = (1 - omega) * x_prev[i] + omega / A[i,i] * (b[i] - first_sum - secondary_sum)
      err = np.linalg.norm(np.abs(x_k - x_prev),2) / np.linalg.norm(x_k, 2)
      x_prev = np.array(x_k)

  return (x_k, num_iterations, err)

A_matrix = generateA([-1.0, 4.0, -1.0], [-1.0, 0.0, 1.0])

### error below 1e-6 ###

#Jacobi Values
soln_tuple_jacobi = Jacobi(A_matrix, b, 1.e-6)
print "\nJacobi Solution Vector (below 1e-6):\n{0}\n".format(soln_tuple_jacobi[0])
print "Iteration Required to reach this solution (below 1e-6): {0}\n".format(soln_tuple_jacobi[1])
print "Relative Error of Our Iteration Method (below 1e-6): {:6}\n".format(soln_tuple_jacobi[2])
print "Absolute Error of Our Iteration From Hw4 (below 1e-6): {:6}\n".format(5.43822775257e-07)

#Gauss Seidel Values
soln_tuple_gauss_seidel = Gauss_Seidel(A_matrix, b, 1.e-6)
print "\nGauss Seidel Solution Vector (below 1e-6):\n{0}\n".format(soln_tuple_gauss_seidel[0])
print "Iteration Required to reach this solution (below 1e-6): {0}\n".format(soln_tuple_gauss_seidel[1])
print "Relative Error of Our Iteration Method (below 1e-6): {:6}\n".format(soln_tuple_gauss_seidel[2])
print "Absolute Error of Our Iteration From Hw4 (below 1e-6): {:6}\n".format(6.76012150997e-07)

# SOR Values
soln_tuple_sor = SOR(A_matrix, b, 1.1, 1.e-6)
print "\nSOR Solution Vector (below 1e-6):\n{0}\n".format(soln_tuple_sor[0])
print "Iteration Required to reach this solution (below 1e-6): {0}\n".format(soln_tuple_sor[1])
print "Relative Error of Our Iteration Method (below 1e-6): {:6}\n".format(soln_tuple_sor[2])
print "Absolute Error of Our Iteration From Hw4 (below 1e-6): {:6}\n".format(2.91176586773e-07)

### error below 1e-8 ###

#Jacobi Values
soln_tuple_jacobi = Jacobi(A_matrix, b, 1.e-8)
print "\nJacobi Solution Vector (below 1e-8):\n{0}\n".format(soln_tuple_jacobi[0])
print "Iteration Required to reach this solution (below 1e-8): {0}\n".format(soln_tuple_jacobi[1])
print "Relative Error of Our Iteration Method (below 1e-8): {:6}\n".format(soln_tuple_jacobi[2])
print "Absolute Error of Our Iteration From Hw4 (below 1e-6): {:6}\n".format(5.43822775257e-07)

#Gauss Seidel Values
soln_tuple_gauss_seidel = Gauss_Seidel(A_matrix, b, 1.e-8)
print "\nGauss Seidel Solution Vector (below 1e-8):\n{0}\n".format(soln_tuple_gauss_seidel[0])
print "Iteration Required to reach this solution (below 1e-8): {0}\n".format(soln_tuple_gauss_seidel[1])
print "Relative Error of Our Iteration Method (below 1e-8): {:6}\n".format(soln_tuple_gauss_seidel[2])
print "Absolute Error of Our Iteration From Hw4 (below 1e-6): {:6}\n".format(6.76012150997e-07)

# SOR Values
soln_tuple_sor = SOR(A_matrix, b, 1.1, 1.e-8)
print "\nSOR Solution Vector (below 1e-8):\n{0}\n".format(soln_tuple_sor[0])
print "Iteration Required to reach this solution (below 1e-8): {0}\n".format(soln_tuple_sor[1])
print "Relative Error of Our Iteration Method (below 1e-8): {:6}\n".format(soln_tuple_sor[2])


###ANSWERS###

#a)

#i) For each method, how does the absolute error (from Homework 4 with sigma = 10e-6) compare to the relative error?
"""It seems that the absolute error was larger than the relative error in
every iteration method, except for in the case of the Gauss Seidel solver. I'm
not sure why the Gauss Seidel method would exhibit this anomalous behavior with
respect to the final error."""

#ii) Which method required the fewest iterations?
"""The SOR method required the fewest number of iterations in both the error = 10e-6
and error = 10e-8 cases."""

#iii) What do you observe about reaching a tighter convergence tolerance?
"""A tighter convergence tolerance correlates with a larger number of iterations
to reach our solution. This makes sense given that a lower tolerance translates
to a more accurate solution vector. Satisfying absolute error also seems to require
more iterations than satisfying the same amount of relative error. """

#b) Perform an experiment to determine the optimal omega for SOR.
#   Explain your procedure and include the results.
"""We can find an optimal omega value for SOR by trying several different
values with our SOR solver and seeing which attempt resulted in the fewest
number of iterations to achieve a sufficiently accurate solution. A plot of
these values is produced below. Omega = 1.06072144289 was found to be the
optimal value according to the results of this experiment. """

omega_attempts = np.linspace(0.1, 1.8, 500)
num_iterations = []
omega_to_itr = {}
for k in range(len(omega_attempts)):
  soln_tuple_sor = SOR(A_matrix, b, omega_attempts[k], 1.e-6)
  num_iterations.append(soln_tuple_sor[1])
  omega_to_itr[omega_attempts[k]] = soln_tuple_sor[1]

print "The optimal Omega value produced in experiment was: {0}".format(min(omega_to_itr, key = omega_to_itr.get))
plot(omega_attempts, num_iterations, 'r')
title("Omega Values of SOR vs Iteration Number (for error = 1e-6)")
xlabel("Inputted Value of Omega")
ylabel("Number of Iterations")
show()




