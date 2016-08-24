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
# from cvxpy import *
import math
import argparse

parser = argparse.ArgumentParser("Fourier representation of multivalue density")
parser.add_argument("-d","--data", help = "the name of sample data", type=str, default="insurance")
parser.add_argument("-s","--sample", help = "the number of samples", type=int, default=1000)
parser.add_argument("-sgt","--sampleGT", help = "the number of samples for ground truth", type=int, default=10000)
parser.add_argument("-k","--parameter", help = " the hyperparameter to decide the order of fourier coefficients", type=int, default=0)
args = parser.parse_args()

print "loading data ( %s )..." % args.data
dataRaw = np.loadtxt("data/" + args.data + "_" + str(args.sample) + ".csv", delimiter=",", skiprows=1)
dataRawGT = np.loadtxt("data/" + args.data + "_" + str(args.sampleGT) + ".csv", delimiter=",", skiprows=1)

# sanity check
# v = np.array([2,3,4,6])
# nn = 100
# dataRaw = np.random.randint(10,size=[nn+1,v.size]) % v

v = dataRaw[0].astype('int')
data = dataRaw[1:].astype('int')
dataGT = dataRawGT[1:].astype('int')
m, n = data.shape
N = np.prod(v)

# f = np.zeros(N)
# bases = np.array([72,24,6,1])
# for i in range(data.shape[0]):
#     f[bases.dot(data[i])] += 1
# f /= float(nn)

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
vPerm = {}
vNum = []
k = 0
for i in np.sort(list(set(vNew))):
    vPerm[i] = np.where(vNew==i)[0]
    vNum.append(vPerm[i].size)
    dataPrimePerm[:,k:k+vPerm[i].size] = dataPrime[:,vPerm[i]]
    vPrimePerm[k:k+vPerm[i].size] = vNew[vPerm[i]]
    k += vPerm[i].size


# New k (representing the order of coefficients to keep)
k = args.parameter
p = np.max(vNum) - k + 1

print "computing indexes with largest probabilities ..."
idx_list = utils_valid.idx_sort(dataPrimePerm, m)
A = utils_valid.build_A(p,n,idx_list)
A = (A[1:] - A[0]) % vPrimePerm

print vPrimePerm

k = 0
vPos = {}
for i in np.sort(list(set(vNew))):
    print i
    Atemp = A[:,k:k+vPerm[i].size]
    print 'gaussian elimination ...'
    A_ge, Perm = utils_valid.gaussianEliminationGeneral(Atemp, i)
    print 'building H ...'
    H = utils_valid.solutionHGeneral(A_ge, Perm, i)
    print 'finding fourier postitions ...'
    pos = utils_valid.fourierCoeffPosition(H, i)
    k += vPerm[i].size
    vPos[i] = pos

# Computing Fourier Coefficients given data and positions
coeffMatrix = {}
k = 0
for i in np.sort(list(set(vNew))):
    print i
    dataTemp = dataPrimePerm[:,k:k+vPerm[i].size]
    coeffMatrix[i] = utils_valid.computeFourierCoefficientMatrix(vPos[i], dataTemp, i)
    k += vPerm[i].size

print coeffMatrix

IPython.embed()

# Inference
# Random marginal assignment
np.random.seed(10)
# prob = np.zeros(N, dtype='cfloat')
# H = utils_valid.generalHadamard(v)
probTwo = np.zeros([1000,2])
for i in range(1000):
    # assignment = utils_valid.expansion(i,v)
    # assignment = utils_valid.randomAssignment(v)
    assignment = dataGT[np.random.randint(m)]
    assignment[np.where(np.random.randint(2,size=v.size)==0)] = -1
    print assignment
    vAssign = utils_valid.decomposeData(assignment, vNew, vPrime, vPerm)
    prob = utils_valid.marginalizedInference(vAssign, coeffMatrix, vPos) / N
    print prob
    prob2 = utils_valid.marginalizedInferenceEmpirical(assignment, dataGT)
    probTwo[i,0] = np.real(prob)
    probTwo[i,1] = prob2


# a = np.conjugate(H).dot(H).dot(f) / N
x = np.real(probTwo[:,0])
y = probTwo[:,1]
# x[np.where(x<0)] == 0
# y[np.where(y<0)] == 0

diff = np.abs(x-y) / y
print np.mean(diff)

# non zeros probabilities inference




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


