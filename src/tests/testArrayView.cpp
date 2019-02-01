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


#include "gtest/gtest.h"
#include <string>

#include "Array.hpp"
#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include "testUtils.hpp"
#include <type_traits>

#ifdef USE_CUDA

#ifdef USE_CHAI
#include "chai/util/forall.hpp"
#endif

#endif

namespace LvArray
{

template < typename T >
using array = Array< T, 1, int >;

template < typename T >
using arrayView = ArrayView< T, 1, int > const;

template < typename T >
using arrayView_nc = ArrayView< T, 1, int >;

template < typename T >
using arraySlice = ArraySlice< T, 1, int > const;


template < typename T >
using array2D = Array< T, 2, int >;

template < typename T >
using arrayView2D = ArrayView< T, 2, int > const;

template < typename T >
using array3D = Array< T, 3, int >;

template < typename T >
using arrayView3D = ArrayView< T, 3, int > const;

namespace internal
{


template < typename T >
void test2DAccessors( arrayView2D< T > & v )
{
  const int I = v.size(0);
  const int J = v.size(1);
  
  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      v[ i ][ j ] = T( J * i + j );
    }
  }

  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      v( i, j ) *= T( 2 );
    }
  }

  for( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      T val( J * i + j  );
      val *= T( 2 );
      EXPECT_EQ( v[ i ][ j ], val );
    }
  }
}


template < typename T >
void test3DAccessors( arrayView3D< T > & v )
{
  const int I = v.size(0);
  const int J = v.size(1);
  const int K = v.size(2);
  
  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      for ( int k = 0; k < K; ++k )
      {
        v[ i ][ j ][ k ] = T( J * K * i + K * j + k );
      }
    }
  }

  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      for ( int k = 0; k < K; ++k )
      {
        v( i, j, k ) *= T( 2 );
      }
    }
  }

  for( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      for ( int k = 0; k < K; ++k )
      {
        T val( J * K * i + K * j + k );
        val *= T( 2 );
        EXPECT_EQ( v[ i ][ j ][ k ], val );
      }
    }
  }
}


#ifdef USE_CUDA


template < typename T >
void testMemoryMotion( arrayView< T > & v )
{
  const int N = v.size();
  
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
void testMemoryMotionConst( array< T > & a )
{
  const int N = a.size();

  for ( int i = 0; i < N; ++i )
  {
    a[ i ] = T( i );
  }

  arrayView< T const > & v = a.toViewConst();

  // Capture the view on the device.
  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      assert(v[i] == T( i ));
    }
  );

  // Change the values that the array holds.
  for ( int i = 0; i < N; ++i )
  {
    a[ i ] = T( 2 * i );
  }

  // Copy the array back to the host, should be a no-op since it was captured as T const
  // and therefore the values should be those set above.
  forall( sequential(), 0, N,
    [=]( int i )
    {
      EXPECT_EQ( v[ i ], T( 2 * i ) );
    }
  );
}


template < typename T >
void testMemoryMotionMove( array< T > & a )
{
  const int N = a.size();
  
  for ( int i = 0; i < N; ++i )
  {
    a[ i ] = T( i );
  }

  a.move( chai::GPU );
  arrayView< T > & v = a;
  T const * const v_device_ptr = v.data();

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= v[ i ];
    }
  );

  /* The copy construction shouldn't have changed the pointer since we moved it. */
  ASSERT_EQ(v_device_ptr, v.data());

  a.move( chai::CPU );
  for ( int i = 0; i < N; ++i )
  {
    T val = T( i );
    val *= val;
    EXPECT_EQ( v[ i ], val );
  }
}

template < typename T >
void testMemoryMotionMultiple( arrayView< T > & v )
{
  const int N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );


  forall( sequential(), 0, N,
    [=]( int i )
    {
      T val( i );
      val *= T( 2 ) * T( 2 ) * T( 2 ) * T( 2 );
      EXPECT_EQ( v[ i ], val );
    }
  );
}

template < typename T >
void testMemoryMotionMultipleMove( array< T > & a )
{
  arrayView< T > & v = a;

  const int N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( int i = 0; i < N; ++i )
  {
    v[ i ] *= T( 2 );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      v[ i ] *= T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( int i = 0; i < N; ++i )
  {
    T val( i );
    val *= T( 2 ) * T( 2 ) * T( 2 ) * T( 2 );
    EXPECT_EQ( v[ i ], val );
  }
}


template < typename T >
void testMemoryMotionArray( array< array< T > > & a )
{
  arrayView< arrayView< T > > & v = a.toView();

  const int N = v.size();

  for ( int i = 0; i < N; ++i )
  { 
    for ( int j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        T val( N * i + j );
        val *= val;
        EXPECT_EQ( v[ i ][ j ], val );
      }
    }
  );
}

template < typename T >
void testMemoryMotionArrayConst( array< array< T > > & a )
{
  const int N = a.size();

  // Create a shallow copy of a that we can modify later.
  array< arrayView_nc< T > > a_copy( N );

  for ( int i = 0; i < N; ++i )
  { 
    a_copy[ i ] = a[ i ];
    for ( int j = 0; j < N; ++j )
    {
      a[ i ][ j ] = T( N * i + j );
    }
  }

  // Create a const view and capture it.  
  arrayView< arrayView< T const > > & v = a.toViewConst();
  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        assert(v[ i ][ j ] == T( N * i + j ));
      }
    }
  );

  // Modify a_copy. We can't use a directly since the inner arrays
  // have device pointers.
  for ( int i = 0; i < N; ++i )
  { 
    for ( int j = 0; j < N; ++j )
    {
      a_copy[ i ][ j ] = T( 2 * ( N * i + j ) );
    }
  }

  // Check that the modifications weren't overwritten.
  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        EXPECT_EQ( v[ i ][ j ], T( 2 * ( N * i + j ) ) );
      }
    }
  );
}


template < typename T >
void testMemoryMotionArrayMove( array< array< T > > & a )
{
  arrayView< arrayView< T > > & v = a.toView();

  const int N = v.size();

  for ( int i = 0; i < N; ++i )
  { 
    for ( int j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  a.move( chai::CPU );
  for ( int i = 0; i < N; ++i )
  {
    for ( int j = 0; j < N; ++j )
    {
      T val( N * i + j );
      val *= val;
      EXPECT_EQ( v[ i ][ j ], val );
    }
  }
}


template < typename T >
void testMemoryMotionArray2( array< array< array< T > > > & a )
{
  arrayView< arrayView< arrayView< T > > > & v = a.toView();

  const int N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {  
    for ( int j = 0; j < N; ++j )
    {
      for ( int k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        for ( int k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        for ( int k = 0; k < N; ++k )
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
void testMemoryMotionArray2Const( array< array< array< T > > > & a )
{
  const int N = a.size();

  // Create a shallow copy of a that we can modify later.
  array< array< arrayView_nc< T > > > a_copy( N );
  
  for ( int i = 0; i < N; ++i )
  { 
    a_copy[ i ].resize(a[i].size());
    for ( int j = 0; j < N; ++j )
    {
      a_copy[ i ][ j ] = a[ i ][ j ];
      for ( int k = 0; k < N; ++k )
      {
        a[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  arrayView< arrayView< arrayView< T const > > > & v = a.toViewConst();

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        for ( int k = 0; k < N; ++k )
        {
          assert(v[ i ][ j ][ k ] == T( N * N * i + N * j + k ));
        }
      }
    }
  );

  // Modify a_copy. We can't use a directly since the inner arrays
  // have device pointers.
  for ( int i = 0; i < N; ++i )
  {
    for ( int j = 0; j < N; ++j )
    {
      for ( int k = 0; k < N; ++k )
      {
        a_copy[ i ][ j ][ k ] = T( 2 * ( N * N * i + N * j + k ) );
      }
    }
  }

  // Check that the modifications weren't overwritten.
  forall( sequential(), 0, N,
    [=]( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        for ( int k = 0; k < N; ++k )
        {
          EXPECT_EQ( v[ i ][ j ][ k ], T( 2 * ( N * N * i + N * j + k ) ) );
        }
      }
    }
  );
}


template < typename T >
void testMemoryMotionArrayMove2( array< array< array< T > > > & a )
{
  arrayView< arrayView< arrayView< T > > > & v = a.toView();
  
  const int N = v.size();
  
  for ( int i = 0; i < N; ++i )
  {  
    for ( int j = 0; j < N; ++j )
    {
      for ( int k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < N; ++j )
      {
        for ( int k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  a.move( chai::CPU );
  for ( int i = 0; i < N; ++i )
  {  
    for ( int j = 0; j < N; ++j )
    {
      for ( int k = 0; k < N; ++k )
      {
        T val( N * N * i + N * j + k );
        val *= val;
        EXPECT_EQ( v[ i ][ j ][ k ], val );
      }
    }
  }
}


template < typename T >
void test2DAccessorsDevice( arrayView2D< T > & v )
{
  const int I = v.size(0);
  const int J = v.size(1);
  
  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      v[ i ][ j ] = T( J * i + j );
    }
  }

  forall( cuda(), 0, I,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < J; ++j )
      {
        v( i, j ) *= T( 2 );
        v[ i ][ j ] *= T( 2 );
      }
    }
  );

  forall( sequential(), 0, I,
    [=] ( int i )
    {
      for ( int j = 0; j < J; ++j )
      {
        T val( J * i + j  );
        val *= T( 2 ) * T( 2 );
        EXPECT_EQ( v[ i ][ j ], val );
      }
    }
  );
}


template < typename T >
void test3DAccessorsDevice( arrayView3D< T > & v )
{
  const int I = v.size(0);
  const int J = v.size(1);
  const int K = v.size(2);
  
  for ( int i = 0; i < I; ++i )
  {
    for ( int j = 0; j < J; ++j )
    {
      for ( int k = 0; k < K; ++k )
      {
        v[ i ][ j ][ k ] = T( J * K * i + K * j + k );
      }
    }
  }

  forall( cuda(), 0, I,
    [=] __device__ ( int i )
    {
      for ( int j = 0; j < J; ++j )
      {
        for ( int k = 0; k < K; ++k )
        {
          v( i, j, k ) *= T( 2 );
          v[ i ][ j ][ k ] *= T( 2 );
        }
      }
    }
  );

  forall( sequential(), 0, I,
    [=] ( int i )
    {
      for ( int j = 0; j < J; ++j )
      {
        for ( int k = 0; k < K; ++k )
        {
          T val( J * K * i + K * j + k );
          val *= T( 2 ) * T( 2 );
          EXPECT_EQ( v[ i ][ j ][ k ], val );
        }
      }
    }
  );
}

template < typename T >
void testSizeOnDevice( arrayView3D< T > & v )
{

  forall( cuda(), 0, v.size(0),
    [=] __device__ ( int i )
    {
      const int I = v.size(0);
      const int J = v.size(1);
      const int K = v.size(2);
      for ( int j = 0; j < J; ++j )
      {
        for ( int k = 0; k < K; ++k )
        {
          v( i, j, k ) = T( J * K * i + K * j + k );
        }
      }
    }
  );

  const int I = v.size(0);
  const int J = v.size(1);
  const int K = v.size(2);

  forall( sequential(), 0, I,
    [=] ( int i )
    {
      for ( int j = 0; j < J; ++j )
      {
        for ( int k = 0; k < K; ++k )
        {
          T val( J * K * i + K * j + k );
          EXPECT_EQ( v[ i ][ j ][ k ], val );
        }
      }
    }
  );
}


#endif

} /* namespace internal */


TEST( ArrayView, test_upcast )
{
  constexpr int N = 10;

  array< int > arr(N);

  for( int a=0 ; a<N ; ++a )
  {
    arr[a] = a;
  }

  {
    arrayView< int > & arrView = arr;
    arrayView< int const > & arrViewConst = arr;

    for( int a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrView[a] );
      ASSERT_EQ( arr[a], arrViewConst[a] );
    }

    arraySlice<int> arrSlice1 = arrView;
    arraySlice<int const> & arrSlice4 = arrView;

    for( int a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrSlice1[a] );
      ASSERT_EQ( arr[a], arrSlice4[a] );
    }
  }

  {
    arraySlice<int> arrSlice1 = arr;
    arraySlice<int const> & arrSlice4 = arr;

    for( int a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrSlice1[a] );
      ASSERT_EQ( arr[a], arrSlice4[a] );
    }
  }
}


TEST( ArrayView, test_dimReduction )
{
  const int N = 10;
  array2D< int > v( N, 1 );
  arrayView2D<int> & vView = v;
  arrayView<int> & vView1d = v.dimReduce();

  for( int a=0 ; a<N ; ++a )
  {
    v[a][0] = 2*a;
  }

  ASSERT_EQ( vView1d.data(), v.data() );

  for( int a=0 ; a<N ; ++a )
  {
    ASSERT_EQ( vView[a][0], v[a][0] );
    ASSERT_EQ( vView1d[a], v[a][0] );
  }
}


TEST( ArrayView, 2DAccessors )
{
  constexpr int N = 20;
  constexpr int M = 10;

  {
    array2D< int > a( N, M );
    internal::test2DAccessors( a );
  }

  {
    array2D< Tensor > a( N, M );
    internal::test2DAccessors( a );
  }
}


TEST( ArrayView, 3DAccessors )
{
  constexpr int N = 7;
  constexpr int M = 8;
  constexpr int P = 9;

  {
    array3D< int > a( N, M, P );
    internal::test3DAccessors( a );
  }

  {
    array3D< Tensor > a( N, M, P );
    internal::test3DAccessors( a );
  }
}


#ifdef USE_CUDA

CUDA_TEST( ArrayView, memoryMotion )
{
  constexpr int N = 10;

  {
    array< int > a( N );
    internal::testMemoryMotion( a );
  }

  // {
  //   array< Tensor > a( N );
  //   internal::testMemoryMotion( a );
  // }
}

CUDA_TEST( ArrayView, memoryMotionConst )
{
  constexpr int N = 100;

  {
    array< int > a( N );
    internal::testMemoryMotionConst( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionConst( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMove )
{
  constexpr int N = 100;

  {
    array< int > a( N );
    internal::testMemoryMotionMove( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMultiple )
{
  constexpr int N = 100;

  {
    array< int > a( N );
    internal::testMemoryMotionMultiple( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMultiple( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMultipleMove )
{
  constexpr int N = 100;

  {
    array< int > a( N );
    internal::testMemoryMotionMultipleMove( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMultipleMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray )
{
  constexpr int N = 10;

  {
    array< array< int > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArray( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArray( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArrayConst )
{
  constexpr int N = 10;

  {
    array< array< int > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayConst( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayConst( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArrayMove )
{
  constexpr int N = 10;

  {
    array< array< int > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray2 )
{
  constexpr int N = 5;

  {
    array< array< array< int > > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( a );
  }

  {
    array< array< array< Tensor > > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray2Const )
{
  constexpr int N = 5;

  {
    array< array< array< int > > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2Const( a );
  }

  {
    array< array< array< Tensor > > > a( N );
    for ( int i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( int j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2Const( a );
  }
}

// CUDA_TEST( ArrayView, memoryMotionArrayMove2 )
// {
//   constexpr int N = 5;

//   {
//     array< array< array< int > > > a( N );
//     for ( int i = 0; i < N; ++i )
//     {
//       a[ i ].resize( N );
//       for ( int j = 0; j < N; ++j )
//       {
//         a[ i ][ j ].resize( N );
//       }
//     }

//     internal::testMemoryMotionArrayMove2( a );
//   }

//   {
//     array< array< array< Tensor > > > a( N );
//     for ( int i = 0; i < N; ++i )
//     {
//       a[ i ].resize( N );
//       for ( int j = 0; j < N; ++j )
//       {
//         a[ i ][ j ].resize( N );
//       }
//     }

//     internal::testMemoryMotionArrayMove2( a );
//   }
// }

CUDA_TEST( ArrayView, 2DAccessorsDevice )
{
  constexpr int N = 20;
  constexpr int M = 10;

  {
    array2D< int > a( N, M );
    internal::test2DAccessorsDevice( a );
  }

  {
    array2D< Tensor > a( N, M );
    internal::test2DAccessorsDevice( a );
  }
}


CUDA_TEST( ArrayView, 3DAccessorsDevice )
{
  constexpr int N = 7;
  constexpr int M = 8;
  constexpr int P = 9;

  {
    array3D< int > a( N, M, P );
    internal::test3DAccessorsDevice( a );
  }

  {
    array3D< Tensor > a( N, M, P );
    internal::test3DAccessorsDevice( a );
  }
}

CUDA_TEST( ArrayView, sizeOnDevice )
{
  constexpr int N = 7;
  constexpr int M = 8;
  constexpr int P = 9;

  {
    array3D< int > a( N, M, P );
    internal::testSizeOnDevice( a );
  }

  {
    array3D< Tensor > a( N, M, P );
    internal::testSizeOnDevice( a );
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
  return result;
}
