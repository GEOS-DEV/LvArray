/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/*
 * main.cpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */


#include "arrayAccessorPerformance.hpp"

using namespace LvArray;


double MatrixMultiply_1D( integer_t const num_i,
                          integer_t const num_j,
                          integer_t const num_k,
                          integer_t const ITERATIONS,
                          double const * const  A,
                          double const * const  B,
                          double * const  C )
{
  int64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[ i*num_j+j ] += A[ i*num_k+k ] * B[ k*num_j+j ] + 3.1415 * A[ i*num_k+k ] + 1.61803 * B[ k*num_j+j ];
        }
      }
    }
  }
  int64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}

double MatrixMultiply_1Dr( integer_t const num_i,
                           integer_t const num_j,
                           integer_t const num_k,
                           integer_t const ITERATIONS,
                           double const * const __restrict__  A,
                           double const * const __restrict__ B,
                           double * const __restrict__ C )
{
  int64_t startTime = GetTimeMs64();
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter )
  {
    for( integer_t i = 0 ; i < num_i ; ++i )
    {
      for( integer_t j = 0 ; j < num_j ; ++j )
      {
        for( integer_t k = 0 ; k < num_k ; ++k )
        {
          C[ i*num_j+j ] += A[ i*num_k+k ] * B[ k*num_j+j ] + 3.1415 * A[ i*num_k+k ] + 1.61803 * B[ k*num_j+j ];
        }
      }
    }
  }
  int64_t endTime = GetTimeMs64();
  return ( endTime - startTime ) / 1000.0;
}



#define MATMULT \
  int64_t startTime = GetTimeMs64(); \
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter ) \
  { \
    for( integer_t i = 0 ; i < num_i ; ++i ) \
    { \
      for( integer_t j = 0 ; j < num_j ; ++j ) \
      { \
        for( integer_t k = 0 ; k < num_k ; ++k ) \
        { \
          C[i][j] += A[i][k] * B[k][j] + 3.1415 * A[i][k] + 1.61803 * B[k][j]; \
        } \
      } \
    } \
  } \
  int64_t endTime = GetTimeMs64(); \
  return ( endTime - startTime ) / 1000.0;

#define MATMULT2 \
  int64_t startTime = GetTimeMs64(); \
  for( integer_t iter = 0 ; iter < ITERATIONS ; ++iter ) \
  { \
    for( integer_t i = 0 ; i < num_i ; ++i ) \
    { \
      for( integer_t j = 0 ; j < num_j ; ++j ) \
      { \
        for( integer_t k = 0 ; k < num_k ; ++k ) \
        { \
          C(i,j) += A(i,k) * B(k,j) + 3.1415 * A(i,k) + 1.61803 * B(k,j); \
        } \
      } \
    } \
  } \
  int64_t endTime = GetTimeMs64(); \
  return ( endTime - startTime ) / 1000.0;



double MatrixMultiply_2D_accessor( integer_t const num_i,
                                   integer_t const num_j,
                                   integer_t const num_k,
                                   integer_t const ITERATIONS,
                                   ArrayView<double,2> const A,
                                   ArrayView<double,2> const B,
                                   ArrayView<double,2> C )
{
  MATMULT
}



double MatrixMultiply_2D_accessorRef( integer_t const num_i,
                                      integer_t const num_j,
                                      integer_t const num_k,
                                      integer_t const ITERATIONS,
                                      ArrayView<double,2> const & A,
                                      ArrayView<double,2> const & B,
                                      ArrayView<double,2>& C )
{
  MATMULT
}



double MatrixMultiply_2D_accessorPBV2( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       ArrayView<double,2> const  A,
                                       ArrayView<double,2> const  B,
                                       ArrayView<double,2> C )
{
  MATMULT2
}

double MatrixMultiply_2D_accessorRef2( integer_t const num_i,
                                       integer_t const num_j,
                                       integer_t const num_k,
                                       integer_t const ITERATIONS,
                                       ArrayView<double,2> const & A,
                                       ArrayView<double,2> const & B,
                                       ArrayView<double,2>& C )
{
  MATMULT2
}
