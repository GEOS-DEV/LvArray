#!/bin/bash

module load gcc/7.3.0

rm *.o gcc73.x
g++  -std=c++14 -O3     -march=native -Wno-vla -I../src  -I. -c arrayAccessorPerformance.cpp
g++  -std=c++14 -O3     -march=native -Wno-vla -I../src  -I. -c arrayAccessorPerformanceMain.cpp
g++  -std=c++14 -O3     -march=native -o gcc73.x  arrayAccessorPerformanceMain.o arrayAccessorPerformance.o

sleep 1
echo gcc73
for i in `seq 1 10`;
do
  ./gcc73.x    $1 $2 $3 100 2 1
    sleep 1
done
