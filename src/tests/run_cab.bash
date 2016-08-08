#!/bin/bash
. /usr/local/tools/dotkit/init.sh
use gcc-4.9.3p
use clang-3.8.1

g++      -std=c++14 -O3 -I../src -o gcc49.x    arrayAccessorPerformance.cpp
clang++  -std=c++14 -O3 -I../src -o clang38.x  arrayAccessorPerformance.cpp

echo gcc49
for i in `seq 1 10`;
do
    srun -n1 -p pdebug ./gcc49.x   200 400 200 100 2 2
done

echo clang38
for i in `seq 1 10`;
do
    srun -n1 -ppdebug ./clang38.x  200 400 200 100 2 2
done
