/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "benchmarkArrayOfArraysReduceKernels.hpp"

namespace LvArray
{
namespace benchmarking
{

#define REDUCE_KERNEL( INT, N, M, a_ij ) \
  VALUE_TYPE sum = 0; \
  for( INT i = 0; i < N; ++i ) \
  { \
    for( INT j = 0; j < M; ++j ) \
    { \
      sum += a_ij; \
    } \
  } \
  return sum

VALUE_TYPE ArrayOfArraysReduce::reduceFortranArray( ArrayOfArraysT< VALUE_TYPE > const & a )
{ REDUCE_KERNEL( INDEX_TYPE, a.size(), a.sizeOfArray( i ), a( i, j ) ); }

VALUE_TYPE ArrayOfArraysReduce::reduceSubscriptArray( ArrayOfArraysT< VALUE_TYPE > const & a )
{ REDUCE_KERNEL( INDEX_TYPE, a.size(), a.sizeOfArray( i ), a[ i ][ j ] ); }

VALUE_TYPE ArrayOfArraysReduce::reduceFortranView( ArrayOfArraysViewT< VALUE_TYPE const, true > const & a )
{ REDUCE_KERNEL( INDEX_TYPE, a.size(), a.sizeOfArray( i ), a( i, j ) ); }

VALUE_TYPE ArrayOfArraysReduce::reduceSubscriptView( ArrayOfArraysViewT< VALUE_TYPE const, true > const & a )
{ REDUCE_KERNEL( INDEX_TYPE, a.size(), a.sizeOfArray( i ), a[ i ][ j ] ); }

VALUE_TYPE ArrayOfArraysReduce::reduceVector( std::vector< std::vector< VALUE_TYPE > > const & a )
{ REDUCE_KERNEL( std::size_t, a.size(), a[ i ].size(), a[ i ][ j ] ); }

#undef REDUCE_KERNEL

} // namespace benchmarking
} // namespace LvArray
