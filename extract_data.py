import numpy as np
import math as math
import collections
import matplotlib.pyplot as plt

SIZE = 36668
NEW_SIZE = SIZE - 882
CORES = 15
N_KERNELS = 14

def read_speedup():

    times = np.zeros((SIZE,CORES), dtype=float)
    run_order = SIZE*[None]

    fin = open("slambench_fair.tasks", "r")
    for i in range(SIZE):
        for j in range(CORES):
            nr, cores, time, name, _, _ = (fin.readline()).split(",")
            times[i, (int(cores)-1)] = time
            run_order[i] = name

    speedups = np.ones((SIZE,CORES), dtype=float)
    for i in range (SIZE):
        for j in range (1, CORES):
            speedups[i,j]= times[i,0]/times[i,j]
    fin.close()
    return speedups, times, run_order


def read_tb():

    threads = np.zeros((SIZE, 3))
    blocks = np.zeros((SIZE, 3))
    fin = open("tb.out", "r")
    for i in  range(SIZE):
        _, block_tmp = (fin.readline()).split(": ")
        block_x, block_y, block_z = block_tmp.split(" ")

        _, thread_tmp = (fin.readline()).split(": ")
        thread_x, thread_y, thread_z = thread_tmp.split(" ")

        blocks[i] = int(block_x), int(block_y), int(block_z)
        threads[i] = int(thread_x), int(thread_y), int(thread_z)
    fin.close()
    return threads, blocks


def read_ptx():

    count = [dict() for _ in range(14)]
    names = ["" for _ in range(14)]
    fin = open("extract_ptx.ptx", "r")
    line = fin.readline()
    i = -1
    while line:
        if ".visible .entry" in line:
            i += 1
            names[i], _ = line[20:].split("4dim")
        elif line[0] == '\t' and  line[1].isalpha():

            instruction = line[1:].split(' ')[0].split('\t')[0]
            if instruction not in count[i]:
                count[i][instruction] = 1
            else:
                count[i][instruction] += 1

        line = fin.readline()
    fin.close()
    return count, names

def remove_mm_kernel(speedups, times, threads, blocks, run_order):

    new_speedups = np.zeros((NEW_SIZE, CORES))
    new_times = np.zeros((NEW_SIZE, CORES))
    new_threads = np.zeros((NEW_SIZE, 3))
    new_blocks = np.zeros((NEW_SIZE, 3))

    new_run_order = ["" for i in range(NEW_SIZE)]

    new_index = 0
    for i in range(SIZE):
        if "void mm2metersKernel<int=1> " not in run_order[i]:

            new_run_order[new_index] = run_order[i]
            for j in range(3):
                new_blocks[new_index, j] = blocks[i, j]
                new_threads[new_index, j] = threads[i, j]
            for j in range(CORES):
                new_times[new_index, j] = times[i, j]
                new_speedups[new_index, j] = speedups[i, j]

            new_index += 1

    return new_speedups, new_times, new_threads, new_blocks, new_run_order


count, names = read_ptx()
speedups, times, run_order = read_speedup()
threads, blocks = read_tb()

speedups, times, threads, blocks, run_order = remove_mm_kernel(speedups, times, threads, blocks, run_order)

###### Histogram #####

max_speedup= np.ones((NEW_SIZE))
for i in range(NEW_SIZE):
    for j in range (1, CORES):
        if speedups[i,j] > max_speedup[i]:
            max_speedup[i] = speedups[i,j]

largest_speedup = math.ceil(max(max_speedup))

'''
accumulated_time= np.zeros(largest_speedup)
for i in range(SIZE):
    index = math.ceil(max_speedup[i])-1
    accumulated_time[index] += data[i,0]/1000000
print (accumulated_time)

'''

bins = [x for x in range(largest_speedup+1)]
print (bins)

_ = plt.hist(max_speedup, bins=bins, weights=[times[i,0]/1000000 for i in range(NEW_SIZE)])
_ = plt.xlabel('Maximal speedup')
_ = plt.ylabel('Kernel time [s]')
plt.show()

