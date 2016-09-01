from collections import Counter
import numpy as np
import IPython
import math

def is_prime(a):
    return all(a % i for i in xrange(2, a))

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return np.array(factors)

def expansion(i, v):
    assert i < np.prod(v)
    n = v.size
    e = np.zeros(n)
    prod = 1
    for j in range(n-1,-1,-1):
        e[j] = i % v[j]
        i = (i - e[j]) / v[j]
    return e

def idx_sort(train, m):
	data_list = []
	for i in range(m):
		assign_temp = list(train[i,:].astype('int'))
		assign_str = ' '.join(map(str, assign_temp))
		data_list.append(assign_str)
	data_counter = Counter(data_list)
	idx = data_counter.most_common()
	return idx

def build_A(p,n,idx_need):
	A = np.zeros([p,n])
	for i in range(p):
		key, val = idx_need[i]
		num = [int(char) for char in key.split()]
		num = np.array(num)
		# num_bar = (num == 0).astype('int')
		# A[i,:] = num_bar
                A[i,:] = num
	A = A.astype('int')
	return A

def gaussianEliminationGeneral(A, v):
    Perm = []
    p = A.shape[0]
    for i in range(p):
        row_keep = np.where(np.sum(A, axis=1) > 0)[0]
        A = A[row_keep]
        if i >= A.shape[0]:
            break
        if np.where(A[i:,i] > 0)[0].size == 0:
            col_swap = np.where(A[i,:] > 0)[0][0]
            Perm.append((i,col_swap))
            col_temp = A[:,i].copy()
            A[:,i] = A[:,col_swap].copy()
            A[:,col_swap] = col_temp.copy()
        elif A[i,i] == 0:
            row_swap = np.where(A[i:,i] > 0)[0][0] + i
            row_temp = A[i,:].copy()
            A[i,:] = A[row_swap,:].copy()
            A[row_swap,:] = row_temp.copy()

        if A[i,i] != 1:
            # Fermit little thm
            A[i,:] = A[i,:]**(v-1) % v

        for j in np.where(A[:,i] > 0)[0]:
            if j != i:
                A[j,:] = (A[j,:] - A[i,:] * A[j,i] % v) % v
        # A = A[:p,:]
    return A, Perm

def solutionHGeneral(A, Perm, v):
    G = (-A[:,A.shape[0]:]) % v
    H = np.append(G,np.eye(G.shape[1]), axis = 0).astype('int')
    Perm.reverse()
    if Perm != []:
        for pair in Perm:
            i,j = pair
            row_temp = H[i,:].copy()
            H[i,:] = H[j,:].copy()
            H[j,:] = row_temp.copy()
    return H

def fourierCoeffPosition(H, v):
    posF = np.zeros([v**(H.shape[1]),H.shape[0]], dtype='int')
    for k in range(v**(H.shape[1])):
        if v == 2:
            a = "{0:b}".format(k)
            a = a.zfill(H.shape[1])
            a = [int(char) for char in str(a)]
            a = np.array(a)
            # print H.dot(a) % v
        else:
            a = expansion(k, np.array([v]*H.shape[1]))
        posF[k,:] = H.dot(a) % v
    return posF

def computeFourierCoefficientMatrix(pos, data, v):
    m, n = data.shape
    numCoeff = pos.shape[0]
    root = math.cos(2*math.pi/v) + 1j * math.sin(2*math.pi/v)
    coeff = np.zeros([numCoeff, m], dtype='complex64')
    for i in range(numCoeff):
        posTemp = pos[i,:]
        coeff[i,:] = root ** (posTemp.dot(data.T))
    return coeff

def randomAssignment(v):
    var = np.random.randint(2, size = v.size)
    value = np.random.randint(np.max(v), size = v.size) % v
    value[np.where(var==0)] = -1
    return value

def decomposeData(assignment, vNew, vPrime, vPerm):
    dataValue = -np.ones(vNew.size, dtype='int')
    k = 0
    for j in range(len(vPrime)):
        if vPrime[j].size > 1:
            if assignment[j] == -1:
                dataValue[k:k+vPrime[j].size] = -np.ones(vPrime[j].size, dtype='int')
            else:
                dataValue[k:k+vPrime[j].size] = expansion(assignment[j],vPrime[j])
        else:
            dataValue[k] = assignment[j]
        k += vPrime[j].size

    k = 0
    vAssign = {}
    dataValuePerm = -np.ones(vNew.size, dtype='int')
    for i in vPerm.keys():
        vAssign[i] = dataValue[vPerm[i]]
        dataValuePerm[k:k+vPerm[i].size] = dataValue[vPerm[i]]
        k += vPerm[i].size

    return vAssign

def customizedTensor(M1, M2):
    assert M1.shape[1] == M2.shape[1]
    M = np.zeros([M1.shape[0] * M2.shape[0], M1.shape[1]], dtype='cfloat')
    k = 0
    for i in range(M1.shape[0]):
        for j in range(M2.shape[0]):
            M[k] = M1[i] * M2[j]
            k += 1
    return M

def marginalizedInference(vAssign, coeffMatrix, vPos):
    coeffMatrixTemp = {}
    for i in vAssign.keys():
        m = coeffMatrix[i].shape[1]
        marginalized = np.where(vAssign[i]<0)[0]
        chosen = np.where(vAssign[i]>=0)[0]
        vPosTemp = np.where( np.sum(vPos[i][:,marginalized], axis=1) == 0 )[0]
        # coeffMatrixTemp[i] = coeffMatrix[i][vPosTemp,:]
        posVar = vPos[i][vPosTemp][:,chosen]
        assign = vAssign[i][chosen]

        root = math.cos(2*math.pi/i) + 1j * math.sin(2*math.pi/i)
        coeffMatrixTemp[i] = (coeffMatrix[i][vPosTemp,:].T * np.conjugate(root ** (posVar.dot(assign) % i)) * (i ** marginalized.size)).T

    M = coeffMatrixTemp[i]
    for i in vAssign.keys()[:-1]:
        M = customizedTensor(M,coeffMatrixTemp[i])

    avg = np.mean(M,axis=1)
    return np.sum(avg)

def marginalizedInferenceEmpirical(assignment, data):
    chosen = np.where(assignment >= 0)[0]
    dataChosen = data[:,chosen]
    diff = np.abs(dataChosen - assignment[chosen])
    return np.where(np.sum(diff, axis=1)==0)[0].size / float(data.shape[0])

###################################################################


def gaussianElimination(A):
	Perm = []
	p = A.shape[0]
	for i in range(p):
		row_keep = np.where(np.sum(A, axis=1) > 0)[0]
		A = A[row_keep]
		if i >= A.shape[0]:
			break
		if np.where(A[i:,i] == 1)[0].size == 0:
			col_swap = np.where(A[i,:] == 1)[0][0]
			Perm.append((i,col_swap))
			col_temp = A[:,i] + 0
			A[:,i] = A[:,col_swap] + 0
			A[:,col_swap] = col_temp + 0
		elif A[i,i] == 0:
			row_swap = np.where(A[i:,i] == 1)[0][0] + i
			row_temp = A[i,:] + 0
			A[i,:] = A[row_swap,:] + 0
			A[row_swap,:] = row_temp + 0
		for j in np.where(A[:,i] == 1)[0]:
			if j != i:
				A[j,:] = A[j,:] ^ A[i,:]
	A = A[:p,:]
	return A, Perm

def solutionH(A, Perm):
	G = A[:,A.shape[0]:]
	H = np.append(G,np.eye(G.shape[1]), axis = 0).astype('int8')
	Perm.reverse()
	if Perm != []:
		for pair in Perm:
			i,j = pair
			row_temp = H[i,:] + 0
			H[i,:] = H[j,:] + 0
			H[j,:] = row_temp + 0
	return H

def posInFourier(H):
	posF = np.zeros([2**(H.shape[1]),H.shape[0]])
	for k in range(2**(H.shape[1])):
		a = "{0:b}".format(k)
		a = a.zfill(H.shape[1])
		a = [int(char) for char in str(a)]
		a = np.array(a)
		# print H.dot(a) % 2
		posF[k,:] = H.dot(a) % 2
	return posF

def coefficients(posF, train):
	coeff = np.zeros(posF.shape[0])
	for i in range(posF.shape[0]):
		pos = posF[i,:]
		pos_x = np.where(pos==1)
		e = 1 - train[:,pos_x][:,0,:]
		mul = np.sum(e,axis=1) % 2
		mul = -2 * mul + 1
		coeff[i] = np.mean(mul) * 1.0 / (2**(posF.shape[1] / 2.0))
	return coeff

def inference(assign, posF, coeff):
	pos_assign = posF * assign
	pos_assign[np.where(pos_assign!=-1)] = 0
	nums = np.sum(pos_assign,axis=1)
	form = np.ones_like(nums)
	form[np.where(nums%2==1)] = -1
	form[0] = 1
	# nums[np.where(nums%2==1)] = -1
	# nums[np.where(nums%2==0)] = 1
	# nums[0] = 1
	prob = form.dot(coeff)
	return prob

def inferenceVectorized(assignVec, posVec, coeff):
	pos_assign = posVec * assignVec
	pos_assign[np.where(pos_assign!=-1)] = 0
	nums = np.sum(pos_assign,axis=2)
	form = np.ones_like(nums)
	form[np.where(nums%2==1)] = -1
	prob = form.transpose((1,0)).dot(coeff)
	return prob


def inferenceCount(data, target, evidenceT, condition, evidenceC):
        var = target + condition
        # print var
        value = evidenceT + evidenceC
        # print value
        dataVar = data[:,var]
        countVar = np.where(np.sum(dataVar == np.array(value), axis=1) == len(var))[0].size
        # print countVar
        countCon = np.where(np.sum(data[:,condition] == np.array(evidenceC), axis=1) == len(condition))[0].size
        # print countCon
        if countVar == 0:
            prob = 0
        else:
            # print countVar
            # print countCon
            prob = countVar / float(countCon)

        print 'prob( x' +  str(target) + '=' + str(evidenceT) + \
                ' | x' + str(condition) + '=' + str(evidenceC) + ' ) = ' + str(prob)

        return prob


def inferenceFourierInner(coeff, pos, var, evidence):
        n = pos.shape[1]
        marginalized = list( set(range(n)) - set(var) )
        posT = np.where( np.sum(pos[:,marginalized], axis=1) == 0 )[0]
        coeffVar = coeff[posT]
        posVar = pos[:,var][posT,:]
        evidence = np.array(evidence) * 2 - 1
        prob = inference(np.array(evidence), posVar, coeffVar) * (2**len(marginalized))
        return prob


def inferenceFourier(data, target, evidenceT, condition, evidenceC):

        coeff, pos = data

        var = target + condition
        value = evidenceT + evidenceC
        probVar = inferenceFourierInner(coeff, pos, var, value)
        probCon = inferenceFourierInner(coeff, pos, condition, evidenceC)

        if probVar == 0:
            prob = 0
        else:
            prob = probVar / float(probCon)

        print 'prob( x' +  str(target) + '=' + str(evidenceT) + \
                ' | x' + str(condition) + '=' + str(evidenceC) + ' ) = ' + str(prob)

        return prob



def generalHadamard(v):

    n = v.shape[0]
    root = np.zeros(n, dtype='cfloat')
    for i in range(n):
        r = math.cos(2*math.pi/v[i]) + 1j * math.sin(2*math.pi/v[i])
        root[i] = r

    N = np.prod(v)
    H = np.zeros([N,N], dtype='cfloat')
    for i in range(N):
        i_expansion = expansion(i,v)
        for j in range(N):
            j_expansion = expansion(j,v)
            power = i_expansion * j_expansion
            H[i][j] = np.prod(root ** power)
    return H
