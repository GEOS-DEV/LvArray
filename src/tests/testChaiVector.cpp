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

#include "gtest/gtest.h"
#include "ChaiVector.hpp"
#include "testUtils.hpp"
#include "Logger.hpp"
#include <vector>

#ifdef USE_CUDA

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

namespace LvArray
{

using size_type = ChaiVector< int >::size_type;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#endif

namespace internal
{

/**
 * @brief Check that the ChaiVector is equivalent to the std::vector. Checks equality using the
 * operator[], the iterator interface, and the raw pointer.
 * @param [in] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to check against.
 */
template < class T >
void compare_to_reference( const ChaiVector< T >& v, const std::vector< T >& v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for( size_type i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], v_ref[ i ] );
  }

  const T* v_ptr = v.data();
  const T* ref_ptr = v_ref.data();
  for( size_type i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v_ptr[ i ], ref_ptr[ i ] );
  }
}

/**
 * @brief Test the push_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of values to append.
 * @return the std::vector compared against.
 */
template < class T >
std::vector< T > push_back_test( ChaiVector< T >& v, int n )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  for( int i = 0 ; i < n ; ++i )
  {
    const T& val = T( i );
    v.push_back( val );
    v_ref.push_back( val );
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the push_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number iterations of appends.
 * @return the std::vector compared against.
 */
template < class T >
std::vector< T > push_back_multiple_test( ChaiVector< T >& v, int n )
{
  EXPECT_TRUE( v.empty() );

  constexpr int M = 10;
  T local_values[M];

  std::vector< T > v_ref;
  int total_values = 0;
  for ( int i = 0 ; i < n ; ++i )
  {
    for ( int n_vals = 1; n_vals < M; ++n_vals )
    {
      for ( int j = 0; j < n_vals; ++j )
      {
        local_values[ j ] = T( total_values++ );
      }

      v.push_back( local_values, n_vals );
      v_ref.insert( v_ref.end(), local_values, local_values + n_vals );
    }
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the insert method of the ChaiVector by inserting one value at a time.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of values to insert.
 * @return the std::vector compared against.
 */
template < class T >
std::vector< T > insert_test( ChaiVector< T >& v, int n )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  for( int i = 0 ; i < n ; ++i )
  {
    const T& val = T( i );
    if( i % 3 == 0 )        /* Insert at the beginning. */
    {
      v.insert( 0, val );
      v_ref.insert( v_ref.begin(), val );
    }
    else if( i % 3 == 1 )   /* Insert at the end. */
    {
      v.insert( v.size(), val );
      v_ref.insert( v_ref.end(), val );
    }
    else                    /* Insert in the middle. */
    {
      v.insert( v.size() / 2, val );
      v_ref.insert( v_ref.begin() + v_ref.size() / 2, val );
    }
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the insert method of the ChaiVector by inserting multiple values at once.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of insertions to do.
 * @param [in] m the number of values to insert per iteration.
 * @return the std::vector compared against.
 */
template < class T >
std::vector< T > insert_multiple_test( ChaiVector< T >& v, int n, int m )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_insert;
  std::vector< T > v_ref;
  for( int i = 0 ; i < n ; ++i )
  {
    v_insert.clear();
    for( int j = 0 ; j < m ; ++j )
    {
      v_insert.push_back( T( m * i + j ) );
    }

    if( i % 3 == 0 )    /* Insert at the beginning. */
    {
      v.insert( 0, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin(), v_insert.begin(), v_insert.end() );
    }
    else if( i % 3 == 1 )   /* Insert at the end. */
    {
      v.insert( v.size(), v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.end(), v_insert.begin(), v_insert.end() );
    }
    else  /* Insert in the middle. */
    {
      v.insert( v.size() / 2, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin() + v_ref.size() / 2, v_insert.begin(), v_insert.end() );
    }
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the erase method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to compare against.
 */
template < class T >
void erase_test( ChaiVector< T >& v, std::vector< T >& v_ref )
{
  const int n_elems = v.size();
  for( int i = 0 ; i < n_elems ; ++i )
  {
    if( i % 3 == 0 )    /* erase the beginning. */
    {
      v.erase( 0 );
      v_ref.erase( v_ref.begin() );
    }
    else if( i % 3 == 1 )   /* erase at the end. */
    {
      v.erase( v.size() - 1 );
      v_ref.erase( v_ref.end() - 1 );
    }
    else  /* erase the middle. */
    {
      v.erase( v.size() / 2 );
      v_ref.erase( v_ref.begin() + v_ref.size() / 2 );
    }

    if( i % 10 == 0 )
    {
      compare_to_reference( v, v_ref );
    }
  }

  ASSERT_TRUE( v.empty() );
  compare_to_reference( v, v_ref );
}

/**
 * @brief Test the pop_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to compare against.
 */
template < class T >
void pop_back_test( ChaiVector< T >& v, std::vector< T >& v_ref )
{
  const int n_elems = v.size();
  for( int i = 0 ; i < n_elems ; ++i )
  {
    v.pop_back();
    v_ref.pop_back();

    if( i % 10 == 0 )
    {
      compare_to_reference( v, v_ref );
    }
  }

  ASSERT_TRUE( v.empty() );
  compare_to_reference( v, v_ref );
}

/**
 * @brief Test the resize method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 */
template < class T >
void resize_test( ChaiVector< T >& v, int n )
{
  ASSERT_TRUE( v.empty() );

  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T* data_ptr = v.data();
  for( int i = 0 ; i < n / 2 ; ++i )
  {
    ASSERT_EQ( data_ptr[ i ], T() );
    data_ptr[ i ] = T( i );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resize( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for( int i = 0 ; i < n / 4 ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }

  v.resize( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for( int i = 0 ; i < n ; ++i )
  {
    v[ i ] = T( 2 * i );
  }

  for( int i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ], T( 2 * i ) );
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 */
template < class T >
void reserve_test( ChaiVector< T >& v, int n )
{
  ASSERT_TRUE( v.empty() );

  v.reserve( n / 2 );

  ASSERT_EQ( v.size(), 0 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T* data_ptr = v.data();
  for( int i = 0 ; i < n / 2 ; ++i )
  {
    v.push_back( T( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.reserve( n );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n );

  for( int i = 0 ; i < n / 2 ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }

  data_ptr = v.data();
  for( int i = n / 2 ; i < n ; ++i )
  {
    v.push_back( T ( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  for( int i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 */
template < class T >
void shallow_copy_test( const ChaiVector< T >& v )
{
  {
    ChaiVector< T > v_cpy( v );

    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.capacity(), v_cpy.capacity() );

    ASSERT_EQ( v.data(), v_cpy.data() );

    for( size_type i = 0 ; i < v.size() ; ++i )
    {
      ASSERT_EQ( v[ i ], v_cpy[ i ] );
      ASSERT_EQ( v[ i ], T( i ) );
    }

    for( size_type i = 0 ; i < v.size() ; ++i )
    {
      v_cpy[ i ] = T( 2 * i );
    }
  }

  for( size_type i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], T( 2 * i ) );
  }
}


#ifdef USE_CUDA

template < typename T >
void testMemoryMotion( ChaiVector< T > & v )
{
  const size_type N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= v[ i ];
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      T val( i );
      val *= val;
      EXPECT_EQ( v[ i ], val );
    }
  );
}

template < typename T >
void testMemoryMotionMove( ChaiVector< T > & v )
{
  const size_type N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= v[ i ];
    }
  );

  v.move(chai::CPU);
  for ( int i = 0; i < N; ++i )
  {
    T val = T( i );
    val *= val;
    EXPECT_EQ( v[ i ], val );
  }
}

template < typename T >
void testMemoryMotionArray( ChaiVector< ChaiVector< T > > & v )
{
  const size_type N = v.size();

  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        T val( N * i + j );
        val *= val;
        EXPECT_EQ( v[ i ][ j ], val );
      }
    }
  );
}

template < typename T >
void testMemoryMotionArrayMove( ChaiVector< ChaiVector< T > > & v )
{
  const size_type N = v.size();

  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  v.move(chai::CPU);
  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      T val( N * i + j );
      val *= val;
      EXPECT_EQ( v[ i ][ j ], val );
    }
  }
}

template < typename T >
void testMemoryMotionArray2( ChaiVector< ChaiVector< ChaiVector< T > > > & v )
{
  const size_type N = v.size();
  
  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      for ( size_type k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        for ( size_type k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        for ( size_type k = 0; k < N; ++k )
        {
          T val( N * N * i + N * j + k );
          val *= val;
          EXPECT_EQ( v[ i ][ j ][ k ], val );
        }
      }
    }
  );
}

template < typename T >
void testMemoryMotionArrayMove2( ChaiVector< ChaiVector< ChaiVector< T > > > & v )
{
  const size_type N = v.size();
  
  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      for ( size_type k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( size_type j = 0; j < N; ++j )
      {
        for ( size_type k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  v.move(chai::CPU);
  for ( size_type i = 0; i < N; ++i )
  {  
    for ( size_type j = 0; j < N; ++j )
    {
      for ( size_type k = 0; k < N; ++k )
      {
        T val( N * N * i + N * j + k );
        val *= val;
        EXPECT_EQ( v[ i ][ j ][ k ], val );
      }
    }
  }
}

#endif

} /* namespace internal */


TEST( ChaiVector, push_back )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::push_back_test( v, N );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_test( v, N );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::push_back_test( v, N );
    v.free();
  }
}

TEST( ChaiVector, push_back_multiple )
{
  constexpr int N = 50;

  {
    ChaiVector< int > v;
    internal::push_back_multiple_test( v, N );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_multiple_test( v, N );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::push_back_multiple_test( v, N );
    v.free();
  }
}

TEST( ChaiVector, insert )
{
  constexpr int N = 300;

  {
    ChaiVector< int > v;
    internal::insert_test( v, N );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::insert_test( v, N );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::insert_test( v, N );
    v.free();
  }
}

TEST( ChaiVector, insert_multiple )
{
  constexpr int N = 100;
  constexpr int M = 10;

  {
    ChaiVector< int > v;
    internal::insert_multiple_test( v, N, M );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::insert_multiple_test( v, N, M );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::insert_multiple_test( v, N, M );
    v.free();
  }
}

TEST( ChaiVector, erase )
{
  constexpr int N = 200;

  {
    ChaiVector< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N );
    internal::erase_test( v, v_ref );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N );
    internal::erase_test( v, v_ref );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    std::vector< TestString > v_ref =  internal::push_back_test( v, N );
    internal::erase_test( v, v_ref );
    v.free();
  }
}

TEST( ChaiVector, pop_back )
{
  constexpr int N = 300;

  {
    ChaiVector< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N );
    internal::pop_back_test( v, v_ref );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N );
    internal::pop_back_test( v, v_ref );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    std::vector< TestString > v_ref =  internal::push_back_test( v, N );
    internal::pop_back_test( v, v_ref );
    v.free();
  }
}

TEST( ChaiVector, resize )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::resize_test( v, N );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::resize_test( v, N );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::resize_test( v, N );
    v.free();
  }
}

TEST( ChaiVector, reserve )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::reserve_test( v, N );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::reserve_test( v, N );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::reserve_test( v, N );
    v.free();
  }
}

TEST( ChaiVector, shallow_copy )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::push_back_test( v, N );
    internal::shallow_copy_test( v );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_test( v, N );
    internal::shallow_copy_test( v );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    internal::push_back_test( v, N );
    internal::shallow_copy_test( v );
    v.free();
  }
}

TEST( ChaiVector, nullptr )
{
  {
    ChaiVector< int > v;
    EXPECT_EQ( v.data(), nullptr );
    v.free();
  }

  {
    ChaiVector< Tensor > v;
    EXPECT_EQ( v.data(), nullptr );
    v.free();
  }

  {
    ChaiVector< TestString > v;
    EXPECT_EQ( v.data(), nullptr );
    v.free();
  }
}

#ifdef USE_CUDA

CUDA_TEST( ChaiVector, memoryMotion )
{
  constexpr int N = 100;

  {
    ChaiVector< int > v( N );
    internal::testMemoryMotion( v );
    v.free();
  }

  {
    ChaiVector< Tensor > v( N );
    internal::testMemoryMotion( v );
    v.free();
  }
}

CUDA_TEST( ChaiVector, memoryMotionMove )
{
  constexpr int N = 100;

  {
    ChaiVector< int > v( N );
    internal::testMemoryMotionMove( v );
    v.free();
  }

  {
    ChaiVector< Tensor > v( N );
    internal::testMemoryMotionMove( v );
    v.free();
  }
}

CUDA_TEST( ChaiVector, memoryMotionArray )
{
  constexpr int N = 10;

  {
    ChaiVector< ChaiVector< int > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
    }

    internal::testMemoryMotionArray( v );
    
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].free();
    }
    v.free();
  }

  {
    ChaiVector< ChaiVector< Tensor > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
    }

    internal::testMemoryMotionArray( v );
    
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].free();
    }
    v.free();
  }
}

CUDA_TEST( ChaiVector, memoryMotionArrayMove )
{
  constexpr int N = 10;

  {
    ChaiVector< ChaiVector< int > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( v );
    
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].free();
    }
    v.free();
  }

  {
    ChaiVector< ChaiVector< Tensor > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( v );
    
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].free();
    }
    v.free();
  }
}

CUDA_TEST( ChaiVector, memoryMotionArray2 )
{
  constexpr int N = 5;

  {
    ChaiVector< ChaiVector< ChaiVector< int > > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( v );
    
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].free();
      }
      v[ i ].free();
    }
    v.free();
  }

  {
    ChaiVector< ChaiVector< ChaiVector< Tensor > > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( v );
    
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].free();
      }
      v[ i ].free();
    }
    v.free();
  }
}

CUDA_TEST( ChaiVector, memoryMotionArrayMove2 )
{
  constexpr int N = 5;

  {
    ChaiVector< ChaiVector< ChaiVector< int > > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArrayMove2( v );
    
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].free();
      }
      v[ i ].free();
    }
    v.free();
  }

  {
    ChaiVector< ChaiVector< ChaiVector< Tensor > > > v( N );
    for ( int i = 0; i < N; ++i )
    {
      v[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArrayMove2( v );
    
    for ( int i = 0; i < N; ++i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ].free();
      }
      v[ i ].free();
    }
    v.free();
  }
}

#endif

} /* namespace LvArray */


int main( int argc, char* argv[] )
{
  logger::InitializeLogger();

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();

#ifdef USE_CHAI
  chai::ArrayManager::finalize();
#endif

  return result;
}
