import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
import utils_valid
from collections import Counter
from numpy.linalg import matrix_rank
import random
import IPython
from scipy.linalg import hadamard
from cvxpy import *
import math
import argparse

parser = argparse.ArgumentParser("Fourier representation of multivalue density")
parser.add_argument("-d","--data", help = "the name of sample data", type=str, default="insurance")
parser.add_argument("-s","--sample", help = "the number of samples", type=int, default=10000)
args = parser.parse_args()

print "loading data ( %s )..." % args.data
dataRaw = np.loadtxt("data/" + args.data + "_" + str(args.sample) + ".csv", delimiter=",", skiprows=1)
v = dataRaw[0].astype('int')
data = dataRaw[1:].astype('int')
m, n = data.shape
N = np.prod(v)

# Data preprocessing ...
# Factorize a number into prime factors
vPrime = {}
for i in range(v.size):
    if utils_valid.is_prime(v[i]) == True:
        vPrime[i] = v[i]
    else:
        temp = utils_valid.prime_factors(v[i])
        vPrime[i] = temp
        n += temp.size - 1

dataPrime = -np.ones([m,n],dtype='int')
for i in range(m):
    k = 0
    for j in range(len(vPrime)):
        if vPrime[j].size > 1:
            dataPrime[i,k:k+vPrime[j].size] = utils_valid.expansion(data[i,j],vPrime[j])
            k += vPrime[j].size
        else:
            dataPrime[i,k] = data[i,j]
            k += 1
assert k == n

vNew = -np.ones(n,dtype='int')
k = 0
for j in range(len(vPrime)):
    vNew[k:k+vPrime[j].size] = vPrime[j]
    k += vPrime[j].size
assert k == n

# Permutation
dataPrimePerm = -np.ones_like(dataPrime, dtype='int')
vPrimePerm = -np.ones_like(vNew, dtype='int')
Vperm = {}
k = 0
for i in np.sort(list(set(vNew))):
    Vperm[i] = np.where(vNew==i)[0]
    dataPrimePerm[:,k:k+Vperm[i].size] = dataPrime[:,Vperm[i]]
    vPrimePerm[k:k+Vperm[i].size] = vNew[Vperm[i]]
    k += Vperm[i].size


# New k (representing the order of coefficients to keep)
k = 10
p = n - k + 1

print "computing indexes with largest probabilities ..."
idx_list = utils_valid.idx_sort(dataPrimePerm, m)
A = utils_valid.build_A(p,n,idx_list)
A = (A[1:] - A[0]) % vPrimePerm

k = 0
for i in np.sort(list(set(vNew))):
    Atemp = A[:,k:k+Vperm[i].size]
    A_ge, Perm = utils_valid.gaussianElimination(Atemp, i)

A_ge, Perm = utils_valid.gaussianElimination(A)

# When N is large, it's impossible to compute the whole Hadamard matrix H
'''
print "building general hadamard matrix ..."
N = np.prod(v)
H = utils_valid.generalHadamard(v)

print "running sanity check ..."
assert H.shape[0] == N
assert H.shape[1] == N
H_CT = np.conjugate(H)
assert np.abs(np.sum(np.abs(H - H.T))) < 1e-8

A = H_CT.dot(H)
R = np.real(A)
I = np.imag(A)

R[np.abs(R) < 1e-8] = 0
I[np.abs(I) < 1e-8] = 0

# print R
# print I

assert np.abs(np.sum(np.abs(I))) < 1e-8
assert np.abs(np.sum(R<0)) < 1e-8
assert np.abs(np.sum(np.abs(H_CT.dot(H) / N - np.eye(N)))) < 1e-8
'''

IPython.embed()


