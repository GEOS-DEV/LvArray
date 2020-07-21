/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
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
