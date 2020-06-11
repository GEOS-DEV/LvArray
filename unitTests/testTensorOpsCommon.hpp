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
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "streamIO.hpp"

// TPL includes
#include <gtest/gtest.h>

#pragma once

namespace LvArray
{
namespace testing
{

template< typename T, int NDIM, int USD, typename ... SIZES >
LVARRAY_HOST_DEVICE
void fill( ArraySlice< T, NDIM, USD > const slice, std::ptrdiff_t offset )
{
  forValuesInSlice( slice, [&offset] ( T & value )
  {
    value = ++offset;
  } );
}

template< typename T, std::ptrdiff_t N >
LVARRAY_HOST_DEVICE
void fill( T ( & array )[ N ], std::ptrdiff_t offset )
{
  for( std::ptrdiff_t i = 0; i < N; ++i )
  {
    array[ i ] = ++offset;
  }
}

template< typename T, std::ptrdiff_t N, std::ptrdiff_t M >
LVARRAY_HOST_DEVICE
void fill( T ( & array )[ N ][ M ], std::ptrdiff_t offset )
{
  for( std::ptrdiff_t i = 0; i < N; ++i )
  {
    for( std::ptrdiff_t j = 0; j < M; ++j )
    {
      array[ i ][ j ] = ++offset;
    }
  }
}

#define CHECK_EQUALITY_1D( N, A, RESULT ) \
  tensorOps::internal::checkSizes< N >( A ); \
  tensorOps::internal::checkSizes< N >( RESULT ); \
  do \
  { \
    for( std::ptrdiff_t _i = 0; _i < N; ++_i ) \
    { PORTABLE_EXPECT_EQ( A[ _i ], RESULT[ _i ] ); } \
  } while ( false )


#define CHECK_EQUALITY_2D( N, M, A, RESULT ) \
  tensorOps::internal::checkSizes< N, M >( A ); \
  tensorOps::internal::checkSizes< N, M >( RESULT ); \
  do \
  { \
    for( std::ptrdiff_t _i = 0; _i < N; ++_i ) \
    { \
      for( std::ptrdiff_t _j = 0; _j < M; ++_j ) \
      { \
        PORTABLE_EXPECT_EQ( A[ _i ][ _j ], RESULT[ _i ][ _j ] ); \
      } \
    } \
  } while ( false )

} // namespace testing
} // namespace LvArray
