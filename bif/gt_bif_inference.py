import bif_parser
from time import time
from prettytable import *
import numpy as np
import argparse
import csv
import IPython

# Perform exact and/or persistent sampling
# inference on a given .bif file,
# showing the time taken and the convergence
# of probability in the case of increasing samples

parser = argparse.ArgumentParser("Exact inference of bayesian network")
parser.add_argument("-d","--network", help = "the name of bayesian network", type=str, default="insurance")
parser.add_argument("-i","--inference", help = "the number of inferences", type=int, default=1000)
parser.add_argument("-e","--evidenceNum", help = "the max number of evidence", type=int, default=5)
args = parser.parse_args()

# Name of .bif file
name = args.network

# (Variable, Value) pair in marginals table to focus on
# key = ('RuggedAuto', 'Football')

start = time()
module_name = bif_parser.parse(name)
print str(time()-start) + "s to parse .bif file into python module"
start = time()
module = __import__(module_name)
print str(time()-start) + "s to import the module"
start = time()
fg = module.create_graph()
print str(time()-start) + "s to create factor graph"
start = time()
bg = module.create_bbn()
print str(time()-start) + "s to create bayesian network"

# Methods of inference to demonstrate
# exact = True
# sampling = True

net = "../data/" + args.network + "_10000.csv"
with open(net, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    k = 1
    for row in reader:
        k += 1
        Variables = row[:]
        if k > 1:
            break

assignment = -1 * np.ones([args.inference, len(Variables)], dtype='int')
p = np.zeros(args.inference)
for j in range(args.inference):
    evidence = {}
    a = assignment[j]
    pos = np.random.randint(len(Variables), size=args.evidenceNum)
    a[pos] = 0
    for i in pos:
        value = bg.domains[Variables[i]]
        idx = np.random.randint(len(value))
        evidence[Variables[i]] = value[idx]
        a[i] = idx

    prob = 1
    for var in evidence.keys():
        key = (var, evidence[var])
        del evidence[var]
        prob *= bg.query(**evidence)[key]
        # print prob
    print prob
    p[j] = prob

outfile = 'Exact_'+args.network+'_'+str(args.evidenceNum)+'_'+str(args.inference)+'.npy'
save = np.concatenate((assignment, p[:,np.newaxis]), axis=1)
np.save(outfile, save)



'''
if exact:
    start = time()
    if not sampling:

        # Set exact=True, sampling=False to
        # just show the exact marginals table
        # and select a key of interest
        bg.q()
    else:
        print 'Exact probability:', bg.query()[key]
    print 'Time taken for exact query:', time()-start

if sampling:
    fg.inference_method = 'sample_db'

    table = PrettyTable(["Number of samples",
                         "Time to generate samples",
                         "Time to query", "Probability",
                         "Difference from previous"])

    for power in range(10):
        n = 2**power
        fg.n_samples = n
        start = time()
        fg.generate_samples(n)
        generate_time = time() - start
        start = time()
        q = fg.query()
        query_time = time() - start
        p = q[key]
        diff = "" if power == 0 else abs(p-prev_p)
        prev_p = p
        table.add_row([n, generate_time, query_time, p, diff])

    print table
'''
