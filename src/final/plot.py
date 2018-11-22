#!/usr/bin/env python
from mpi4py import MPI
import random
import csv
import sys
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
processors = size

csv_file_name = sys.argv[2]

# iterations = 10**7 * 10**(int(sys.argv[1]) - 1)
iter_per_proc = int(sys.argv[1])

height = int(sys.argv[3])
if height % processors != 0:
	raise RuntimeError("Grid with side length of {} is not divisible by number of processors {}".format(height, procesors))
width_per_proc = int(height / processors)
height = width_per_proc * processors
cell_no = height
vector_len = (width_per_proc + 2) * height

def create_grid(square_len, delta):
    cell_no = int(square_len / delta)
    random.seed(23)
    A = np.ndarray(shape=(cell_no, cell_no), dtype=float)
    for i in range(cell_no):
        for j in range(cell_no):
            val = random.random() * square_len / 2
            A.itemset((i, j), val)
    return A
	
def split_grid(A, proc_no):
    if A.shape[0] % proc_no != 0:
        raise RuntimeError
    result = np.split(A, proc_no)
    return result	

def count_next(h_up, h_down, h_left, h_right):
	result = (h_up + h_down + h_left + h_right)/4
	return result
	
def get_cell_value(A, upper_array, lower_array, x, y):
	if x in xrange(A.shape[0]) and y in xrange(A.shape[1]):
		return A[x][y]
		
	if x == -1:
		if y in xrange(A.shape[1]):
			if upper_array.size > 0:
				return upper_array[y]
	elif x == A.shape[0]:
		if y in xrange(A.shape[1]):
			if lower_array.size > 0:
				return lower_array[y]
		
	if y == -1:
		return get_cell_value(A, upper_array, lower_array, x, y+1)
	elif y == A.shape[1]:
		return get_cell_value(A, upper_array, lower_array, x, y-1)
	raise RuntimeError("Get cell value is impossible with params x={}, y={}".format(x,y))
	
def worker_task(A, upper_array, lower_array):
	for i in xrange(A.shape[0]):
		for j in xrange(A.shape[1]):
			up = get_cell_value(A, upper_array, lower_array, i-1, j)
			down = get_cell_value(A, upper_array, lower_array, i+1, j)
			left = get_cell_value(A, upper_array, lower_array, i, j-1)
			right = get_cell_value(A, upper_array, lower_array, i, j+1)
			result = count_next(up, down, left, right)
			A[i][j] = result
	return A

comm.Barrier()
start = MPI.Wtime()

if rank == 0:
	A = create_grid(1, np.float64(1.0/np.float64(cell_no)))
	A_flat = A.flatten()
	# Add upper and lower arrays
	send_data = np.append(np.zeros(height, dtype=np.float64), A_flat)
	send_data = np.append(send_data, np.zeros(height, dtype=np.float64))
	# counts = np.full(processors, vector_len)
	# due to old version of numpy
	counts = np.ones(processors)*vector_len
	displacements = [i*(vector_len - 2*height) for i in xrange(processors)]
else:
	send_data = None
	counts = None
	displacements = None

recv_data = np.empty(vector_len, dtype=np.float64)
sendbuf = [send_data, counts, displacements, MPI.DOUBLE]
comm.Scatterv(sendbuf, recv_data, root=0)

upper_array, array_flat, lower_array = np.split(recv_data, [height, vector_len - height])

grid = array_flat.reshape((width_per_proc, height))

for i in xrange(iter_per_proc):
	if i != 0:
		if rank != 0:
			upper_array = np.empty(height, dtype=np.float64)
			comm.Recv([upper_array, MPI.DOUBLE], source=rank-1, tag=66)
		if rank != processors-1:
			lower_array = np.empty(height, dtype=np.float64)
			comm.Recv([lower_array, MPI.DOUBLE], source=rank+1, tag=77)
			
	grid = worker_task(grid, upper_array, lower_array)
	
	first_row = grid[0]
	last_row = grid[grid.shape[0]-1]
	# print("Proc: {}, Iteration: {}".format(rank, i))
	if rank > 0:
		comm.Isend([first_row, MPI.DOUBLE], dest=rank-1, tag=77)
	if rank < processors-1:
		comm.Isend([last_row, MPI.DOUBLE], dest=rank+1, tag=66)
		
	#send data back to root
	array_flat = grid.flatten()
	A_flat = np.empty(height*height, dtype=np.float64)
	comm.Gather(array_flat, A_flat, root=0)

	if rank == 0:
		A = A_flat.reshape((height, height))
		np.save("./figures/grid_{}".format(i), A)
	comm.Barrier()