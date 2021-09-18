/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "output.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

#pragma once

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< typename T, typename PERMUTATION >
using ArrayT = Array< T, typeManipulation::getDimension< PERMUTATION >, PERMUTATION, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, int NDIM, int USD >
using ArrayViewT = ArrayView< T, NDIM, USD, std::ptrdiff_t, DEFAULT_BUFFER >;


template< typename T, int NDIM, int USD, typename ... SIZES >
LVARRAY_HOST_DEVICE
void fill( ArraySlice< T, NDIM, USD, INDEX_TYPE > const slice, std::ptrdiff_t offset )
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


template< typename T >
std::enable_if_t< std::is_floating_point< T >::value, T >
randomValue( T const maxVal, std::mt19937_64 & gen )
{ return std::uniform_real_distribution< T >( -maxVal, maxVal )( gen ); }


template< typename T >
std::enable_if_t< std::is_integral< T >::value, T >
randomValue( T const maxVal, std::mt19937_64 & gen )
{ return std::uniform_int_distribution< T >( -maxVal, maxVal )( gen ); }


#define CHECK_EQUALITY_1D( N, A, RESULT ) \
  tensorOps::internal::checkSizes< N >( A ); \
  tensorOps::internal::checkSizes< N >( RESULT ); \
  do \
  { \
    for( std::ptrdiff_t _i = 0; _i < N; ++_i ) \
    { PORTABLE_EXPECT_EQ( A[ _i ], RESULT[ _i ] ); } \
  } while ( false )

#define CHECK_NEAR_1D( N, A, RESULT, EPSILON ) \
  tensorOps::internal::checkSizes< N >( A ); \
  tensorOps::internal::checkSizes< N >( RESULT ); \
  do \
  { \
    for( std::ptrdiff_t _i = 0; _i < N; ++_i ) \
    { \
      if( std::is_integral< std::remove_reference_t< decltype( A[ _i ] ) > >::value ) \
      { PORTABLE_EXPECT_EQ( A[ _i ], RESULT[ _i ] ); } \
      else \
      { PORTABLE_EXPECT_NEAR( A[ _i ], RESULT[ _i ], EPSILON ); } \
    } \
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
