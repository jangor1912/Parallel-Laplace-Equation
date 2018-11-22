#!/bin/bash

for j in $(seq 24 24 120)
do
	for i in $(seq 1 8)
	do
		for k in $(seq 0 10)
		do
			if [[ $i -ne 7  &&  $i -ne 5 ]]
			then
				mpiexec -machinefile ./allnodes -np ${i} python ~/AR/assignment1/grid.py 100 result.csv ${j}
			fi
		done
	done
done
