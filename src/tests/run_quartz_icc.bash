#!/bin/bash

module add intel/18.0.0

rm *.o icc18.x
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../src     -c arrayAccessorPerformance.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -I.  -I../src     -c main.cpp
icpc  -std=c++14 -O3     -march=native  -no-opt-matmul -fp-model precise -fp-model source   -o icc18.x  main.o arrayAccessorPerformance.o

sleep 1
#echo icc17
for i in `seq 1 10`;
do
  ./icc18.x  $1 $2 $3 100 2 1
  sleep 1
done
