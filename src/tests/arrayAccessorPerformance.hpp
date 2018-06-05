// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-746361. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the GEOSX Simulation Framework.

//
// GEOSX is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
/*
 * main.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */


#include <sys/time.h>
#include <stdint.h>
#include <string>
#include <math.h>

#include <chrono>
#include <thread>
#include "../src/ArrayView.hpp"

inline uint64_t GetTimeMs64()
{
  struct timeval tv;

  gettimeofday( &tv, NULL );

  uint64_t ret = tv.tv_usec;
  /* Convert from micro seconds (10^-6) to milliseconds (10^-3) */
  ret /= 1000;

  /* Adds the seconds (10^0) after converting them to milliseconds (10^-3) */
  ret += ( tv.tv_sec * 1000 );

  return ret;

}



using namespace multidimensionalArray;

double MatrixMultiply_1D( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const  A,
                          double const * const  B,
                          double * const  C );

double MatrixMultiply_1Dr( integer_t const num_i,
                           integer_t const num_j,
                           integer_t const num_k,
                           integer_t const ITERATIONS,
                           double const * const restrict  A,
                           double const * const restrict B,
                           double * const restrict C );

double MatrixMultiply_2D_accessor( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayView<double,2> const A,
                                   ArrayView<double,2> const B,
                                   ArrayView<double,2> C );


double MatrixMultiply_2D_accessorRef( integer_t const num_i,
                                      integer_t const num_j,
                                      integer_t const num_k,
                                      integer_t const ITERATIONS,
                                      ArrayView<double,2> const & A,
                                      ArrayView<double,2> const & B,
                                      ArrayView<double,2>& C );

double MatrixMultiply_2D_accessorRef2( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       ArrayView<double,2> const & A,
                                       ArrayView<double,2> const & B,
                                       ArrayView<double,2>& C );

double MatrixMultiply_2D_accessorPBV2( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       ArrayView<double,2> const  A,
                                       ArrayView<double,2> const  B,
                                       ArrayView<double,2> C );

double MatrixMultiply_2D_constructAccessorR( integer_t const num_i,
                                             integer_t const num_j,
                                             integer_t const num_k,
                                             integer_t const ITERATIONS,
                                             double const * const __restrict__ ptrA,
                                             integer_t const * const lengthA,
                                             double const * const __restrict__ ptrB,
                                             integer_t const * const lengthB,
                                             double * const __restrict__ ptrC,
                                             integer_t const * const lengthC );
