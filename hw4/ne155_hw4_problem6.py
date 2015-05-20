# NE 155: HW 4, Problem 6
#
#
# @author: Luis Morales

import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg
import sys as sys

### Problem 6 ###

# Defining A using built-in functions (as in problem 4)
values = [-1.0, 4.0, -1.0]

diagonals = [-1.0, 0.0, 1.0]

A_matrix = sp.sparse.diags(values, diagonals, shape = (5, 5)).todense()

b = 100.0 * np.ones(5)

def Jacobi(A, b, n = 5):

    x_k = np.zeros(n)
    x_prev = np.zeros(n)

    err = sys.maxint

    for _ in range(1, 10001):
        if (err < 0.000001):
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
        err = np.linalg.norm(np.abs(x_k - x_prev), 2) 
        x_prev = np.array(x_k)

    return (x_k, num_iterations, err)

def Gauss_Seidel(A, b, n = 5):

    x_k = np.zeros(n)
    x_prev = np.zeros(n)

    err = sys.maxint
    
    for _ in range(1, 10001):
        if (err < 0.000001):
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
        err = np.linalg.norm(np.abs(x_k - x_prev), 2) 
        x_prev = np.array(x_k)

    return (x_k, num_iterations, err)

def SOR(A, b, omega = 1.1, n = 5):

    x_k = np.zeros(n)
    x_prev = np.zeros(n)
    err = sys.maxint
    
    for _ in range(1, 10001):
        if (err < 0.000001):
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
        err = np.linalg.norm(np.abs(x_k - x_prev), 2) 
        x_prev = np.array(x_k)

    return (x_k, num_iterations, err)

#Jacobi Values
soln_tuple_jacobi = Jacobi(A_matrix, b)
print "\nJacobi Solution Vector:\n{0}\n".format(soln_tuple_jacobi[0])
print "Iteration Required to reach this solution: {0}\n".format(soln_tuple_jacobi[1])
print "Absolute Error of Our Iteration Method: {:6}\n".format(soln_tuple_jacobi[2])

#Gauss Seidel Values
soln_tuple_gauss_seidel = Gauss_Seidel(A_matrix, b)
print "\nGauss Seidel Solution Vector\n{0}\n".format(soln_tuple_gauss_seidel[0])
print "Iteration Required to reach this solution: {0}\n".format(soln_tuple_gauss_seidel[1])
print "Absolute Error of Our Iteration Method: {:6}\n".format(soln_tuple_gauss_seidel[2])

# SOR Values
soln_tuple_sor = SOR(A_matrix, b, 1.1)
print "\nSOR Solution Vector\n{0}\n".format(soln_tuple_sor[0])
print "Iteration Required to reach this solution: {0}\n".format(soln_tuple_sor[1])
print "Absolute Error of Our Iteration Method: {:6}\n".format(soln_tuple_sor[2])



