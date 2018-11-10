#!/bin/bash

module add intel/18.0.2

rm *.o icc18.x
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../src     -c arrayAccessorPerformance.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../src     -c arrayAccessorPerformanceMain.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -o icc18.x  arrayAccessorPerformanceMain.o arrayAccessorPerformance.o

sleep 1
echo icc18
for i in `seq 1 10`;
do
  ./icc18.x  $1 $2 $3 100 2 1
  sleep 1
done
