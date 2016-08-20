# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 23:19:28 2016

@author: xiaocheng-mac
"""

import numpy as np
import os

import argparse

parser = argparse.ArgumentParser("Sampling from existing network")
parser.add_argument("-n","--network", help = "the name of the bayesian network", type=str, default="asia")
parser.add_argument("-s","--sample", help = "the number of samples", type=int, default=1000)
args = parser.parse_args()

# print args.degree

networkFileName = args.network + ".bif"
nSample = args.sample
sampleFileName = args.network + "_" + str(nSample) + ".csv"

''' path is the path to the installation folder of gobnilp, please change it here! '''
path = '/Users/chenyue/Documents/16_Summer/gobnilp'

'''sampling using sample.R'''
os.system("Rscript --vanilla sampling.R {} {} {}".format(networkFileName, nSample, sampleFileName))

''' structure learning by gobnlip'''
# os.system(path+'/bin/gobnilp -g=settings_bn_learn.txt -f=dat data/{}'.format(sampleFileName))

'''load the adjacent matrix'''
# adjmat = np.loadtxt("bn.mat")
