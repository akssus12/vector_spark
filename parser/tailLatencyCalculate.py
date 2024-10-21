import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from subprocess import PIPE, Popen

if len(sys.argv) < 2:
    print("Need File to parse!")
    sys.exit(1)
else:
    filename_FIFO = sys.argv[1]
# setupt argument parsing
# usage: python tailLatencyCalculate.py --query [query number]

# setup file to parse: qN.result
path_FIFO = os.path.realpath(filename_FIFO)

# setup new file name: qN_tailLatency
new_fileName = "tailLatency"
put = Popen(["touch", new_fileName], stdin=PIPE, bufsize=-1)
put.communicate()
path1 = os.path.realpath(new_fileName)

list_FIFO = []
latencies_FIFO = []

#list_FAIR = []
#latencies_FAIR = []

#list_EDF = []
#latencies_EDF = []

#list_RDAS = []
#latencies_RDAS = []

with open(path_FIFO,"r") as fp:
    next(fp) # Neglect some initial values
    next(fp)
    next(fp)
    next(fp)
    next(fp)
    for line in fp:
        if "Searching" in line:
            searchLine = line.replace("\n","")
            list_FIFO.append(searchLine.split(" ")[2])

for i in list_FIFO:
    # temp = float(i[1])
    latencies_FIFO.append(i)

cdfx_FIFO = np.sort(latencies_FIFO)
cdfy_FIFO = np.linspace(1 / len(latencies_FIFO), 1.0, len(latencies_FIFO))
#cdfy_FIFO = np.linspace(latencies_FIFO[0], latencies_FIFO[len(latencies_FIFO) - 1], len(latencies_FIFO))

# print(str(1 / len(latencies_FIFO)))

#for latency, prob in zip(cdfx_FIFO, cdfy_FIFO):
#    print(str(latency) + "\t" + str(prob))

a = 0.1
total = 0.0
num = 0

# for tail latencies
tail_25 = np.float64(0.25000000000000000)
tail_50 = np.float64(0.50000000000000000)
tail_95 = np.float64(0.95000000000000000)
tail_99 = np.float64(0.99000000000000000)
tail_99_9 = np.float64(0.99900000000000000)

tmp_25_prob = np.float64(100)
tmp_25_latency = 0

tmp_50_prob = np.float64(100)
tmp_50_latency = 0

tmp_95_prob = np.float64(100)
tmp_95_latency = 0

tmp_99_prob = np.float64(100)
tmp_99_latency = 0

tmp_99_9_prob = np.float64(100)
tmp_99_9_latency = 0

with open(path1, "w") as fp_path:
    for prob, latency in zip(cdfy_FIFO, cdfx_FIFO):
        fp_path.write(str(prob)+ '\t' + str(latency) + '\n')

        if np.abs(tail_25 - prob) < np.abs(tail_25 - tmp_25_prob):
            tmp_25_prob = prob
            tmp_25_latency = latency

        if np.abs(tail_50 - prob) < np.abs(tail_50 - tmp_50_prob):
            tmp_50_prob = prob
            tmp_50_latency = latency

        if np.abs(tail_95 - prob) < np.abs(tail_95 - tmp_95_prob):
            tmp_95_prob = prob
            tmp_95_latency = latency

        if np.abs(tail_99 - prob) < np.abs(tail_99 - tmp_99_prob):
            tmp_99_prob = prob
            tmp_99_latency = latency

        if np.abs(tail_99_9 - prob) < np.abs(tail_99_9 - tmp_99_9_prob):
            tmp_99_9_prob = prob
            tmp_99_9_latency = latency

print(f'{tmp_25_prob}: {tmp_25_latency}')
print(f'{tmp_50_prob}: {tmp_50_latency}')
print(f'{tmp_95_prob}: {tmp_95_latency}')
print(f'{tmp_99_prob}: {tmp_99_latency}')
print(f'{tmp_99_9_prob}: {tmp_99_9_latency}')
