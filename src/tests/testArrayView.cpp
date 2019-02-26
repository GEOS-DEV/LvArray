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

using INDEX_TYPE = std::ptrdiff_t;

template < typename T >
using array = Array< T, 1 >;

template < typename T >
using arrayView = ArrayView< T, 1 > const;

template < typename T >
using arrayView_nc = ArrayView< T, 1 >;

template < typename T >
using arraySlice = ArraySlice< T, 1 > const;


template < typename T >
using array2D = Array< T, 2 >;

template < typename T >
using arrayView2D = ArrayView< T, 2 > const;

template < typename T >
using array3D = Array< T, 3 >;

template < typename T >
using arrayView3D = ArrayView< T, 3 > const;

namespace internal
{


template < typename T >
void test2DAccessors( arrayView2D< T > & v )
{
  const INDEX_TYPE I = v.size(0);
  const INDEX_TYPE J = v.size(1);
  
  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      v[ i ][ j ] = T( J * i + j );
    }
  }

  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      v( i, j ) *= T( 2 );
    }
  }

  for( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
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
  const INDEX_TYPE I = v.size(0);
  const INDEX_TYPE J = v.size(1);
  const INDEX_TYPE K = v.size(2);
  
  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      for ( INDEX_TYPE k = 0; k < K; ++k )
      {
        v[ i ][ j ][ k ] = T( J * K * i + K * j + k );
      }
    }
  }

  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      for ( INDEX_TYPE k = 0; k < K; ++k )
      {
        v( i, j, k ) *= T( 2 );
      }
    }
  }

  for( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      for ( INDEX_TYPE k = 0; k < K; ++k )
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
  const INDEX_TYPE N = v.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= v[ i ];
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
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
  const INDEX_TYPE N = a.size();

  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    a[ i ] = T( i );
  }

  arrayView< T const > & v = a.toViewConst();

  // Capture the view on the device.
  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      assert(v[i] == T( i ));
    }
  );

  // Change the values that the array holds.
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    a[ i ] = T( 2 * i );
  }

  // Copy the array back to the host, should be a no-op since it was captured as T const
  // and therefore the values should be those set above.
  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      EXPECT_EQ( v[ i ], T( 2 * i ) );
    }
  );
}


template < typename T >
void testMemoryMotionMove( array< T > & a )
{
  const INDEX_TYPE N = a.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    a[ i ] = T( i );
  }

  a.move( chai::GPU );
  arrayView< T > & v = a;
  T const * const v_device_ptr = v.data();

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= v[ i ];
    }
  );

  /* The copy construction shouldn't have changed the pointer since we moved it. */
  ASSERT_EQ(v_device_ptr, v.data());

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    T val = T( i );
    val *= val;
    EXPECT_EQ( v[ i ], val );
  }
}

template < typename T >
void testMemoryMotionMultiple( arrayView< T > & v )
{
  const INDEX_TYPE N = v.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );


  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
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

  const INDEX_TYPE N = v.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    v[ i ] = T( i );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    v[ i ] *= T( 2 );
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] *= T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
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

  const INDEX_TYPE N = v.size();

  for ( INDEX_TYPE i = 0; i < N; ++i )
  { 
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
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
  const INDEX_TYPE N = a.size();

  // Create a shallow copy of a that we can modify later.
  array< arrayView_nc< T > > a_copy( N );

  for ( INDEX_TYPE i = 0; i < N; ++i )
  { 
    a_copy[ i ] = a[ i ];
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      a[ i ][ j ] = T( N * i + j );
    }
  }

  // Create a const view and capture it.  
  arrayView< arrayView< T const > > & v = a.toViewConst();
  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        assert(v[ i ][ j ] == T( N * i + j ));
      }
    }
  );

  // Modify a_copy. We can't use a directly since the inner arrays
  // have device pointers.
  for ( INDEX_TYPE i = 0; i < N; ++i )
  { 
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      a_copy[ i ][ j ] = T( 2 * ( N * i + j ) );
    }
  }

  // Check that the modifications weren't overwritten.
  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
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

  const INDEX_TYPE N = v.size();

  for ( INDEX_TYPE i = 0; i < N; ++i )
  { 
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      v[ i ][ j ] = T( N * i + j );
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        v[ i ][ j ] *= v[ i ][ j ];
      }
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    for ( INDEX_TYPE j = 0; j < N; ++j )
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

  const INDEX_TYPE N = v.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {  
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      for ( INDEX_TYPE k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
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
  const INDEX_TYPE N = a.size();

  // Create a shallow copy of a that we can modify later.
  array< array< arrayView_nc< T > > > a_copy( N );
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  { 
    a_copy[ i ].resize(a[i].size());
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      a_copy[ i ][ j ] = a[ i ][ j ];
      for ( INDEX_TYPE k = 0; k < N; ++k )
      {
        a[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  arrayView< arrayView< arrayView< T const > > > & v = a.toViewConst();

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
        {
          assert(v[ i ][ j ][ k ] == T( N * N * i + N * j + k ));
        }
      }
    }
  );

  // Modify a_copy. We can't use a directly since the inner arrays
  // have device pointers.
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      for ( INDEX_TYPE k = 0; k < N; ++k )
      {
        a_copy[ i ][ j ][ k ] = T( 2 * ( N * N * i + N * j + k ) );
      }
    }
  }

  // Check that the modifications weren't overwritten.
  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
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
  
  const INDEX_TYPE N = v.size();
  
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {  
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      for ( INDEX_TYPE k = 0; k < N; ++k )
      {
        v[ i ][ j ][ k ] = T( N * N * i + N * j + k );
      }
    }
  }

  forall( cuda(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] *= v[ i ][ j ][ k ];
        }
      }
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {  
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      for ( INDEX_TYPE k = 0; k < N; ++k )
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
  const INDEX_TYPE I = v.size(0);
  const INDEX_TYPE J = v.size(1);
  
  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      v[ i ][ j ] = T( J * i + j );
    }
  }

  forall( cuda(), 0, I,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        v( i, j ) *= T( 2 );
        v[ i ][ j ] *= T( 2 );
      }
    }
  );

  forall( sequential(), 0, I,
    [=] ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
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
  const INDEX_TYPE I = v.size(0);
  const INDEX_TYPE J = v.size(1);
  const INDEX_TYPE K = v.size(2);
  
  for ( INDEX_TYPE i = 0; i < I; ++i )
  {
    for ( INDEX_TYPE j = 0; j < J; ++j )
    {
      for ( INDEX_TYPE k = 0; k < K; ++k )
      {
        v[ i ][ j ][ k ] = T( J * K * i + K * j + k );
      }
    }
  }

  forall( cuda(), 0, I,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        for ( INDEX_TYPE k = 0; k < K; ++k )
        {
          v( i, j, k ) *= T( 2 );
          v[ i ][ j ][ k ] *= T( 2 );
        }
      }
    }
  );

  forall( sequential(), 0, I,
    [=] ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        for ( INDEX_TYPE k = 0; k < K; ++k )
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
    [=] __device__ ( INDEX_TYPE i )
    {
      const INDEX_TYPE I = v.size(0);
      const INDEX_TYPE J = v.size(1);
      const INDEX_TYPE K = v.size(2);
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        for ( INDEX_TYPE k = 0; k < K; ++k )
        {
          v( i, j, k ) = T( J * K * i + K * j + k );
        }
      }
    }
  );

  const INDEX_TYPE I = v.size(0);
  const INDEX_TYPE J = v.size(1);
  const INDEX_TYPE K = v.size(2);

  forall( sequential(), 0, I,
    [=] ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        for ( INDEX_TYPE k = 0; k < K; ++k )
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
  constexpr INDEX_TYPE N = 10;

  array< INDEX_TYPE > arr(N);

  for( INDEX_TYPE a=0 ; a<N ; ++a )
  {
    arr[a] = a;
  }

  {
    arrayView< INDEX_TYPE > & arrView = arr;
    arrayView< INDEX_TYPE const > & arrViewConst = arr;

    for( INDEX_TYPE a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrView[a] );
      ASSERT_EQ( arr[a], arrViewConst[a] );
    }

    arraySlice<INDEX_TYPE> arrSlice1 = arrView;
    arraySlice<INDEX_TYPE const> & arrSlice4 = arrView;

    for( INDEX_TYPE a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrSlice1[a] );
      ASSERT_EQ( arr[a], arrSlice4[a] );
    }
  }

  {
    arraySlice<INDEX_TYPE> arrSlice1 = arr;
    arraySlice<INDEX_TYPE const> & arrSlice4 = arr;

    for( INDEX_TYPE a=0 ; a<N ; ++a )
    {
      ASSERT_EQ( arr[a], arrSlice1[a] );
      ASSERT_EQ( arr[a], arrSlice4[a] );
    }
  }
}


TEST( ArrayView, test_dimReduction )
{
  const INDEX_TYPE N = 10;
  array2D< INDEX_TYPE > v( N, 1 );
  arrayView2D<INDEX_TYPE> & vView = v;
  arrayView<INDEX_TYPE> & vView1d = v.dimReduce();

  for( INDEX_TYPE a=0 ; a<N ; ++a )
  {
    v[a][0] = 2*a;
  }

  ASSERT_EQ( vView1d.data(), v.data() );

  for( INDEX_TYPE a=0 ; a<N ; ++a )
  {
    ASSERT_EQ( vView[a][0], v[a][0] );
    ASSERT_EQ( vView1d[a], v[a][0] );
  }
}


TEST( ArrayView, 2DAccessors )
{
  constexpr INDEX_TYPE N = 20;
  constexpr INDEX_TYPE M = 10;

  {
    array2D< INDEX_TYPE > a( N, M );
    internal::test2DAccessors( a );
  }

  {
    array2D< Tensor > a( N, M );
    internal::test2DAccessors( a );
  }
}


TEST( ArrayView, 3DAccessors )
{
  constexpr INDEX_TYPE N = 7;
  constexpr INDEX_TYPE M = 8;
  constexpr INDEX_TYPE P = 9;

  {
    array3D< INDEX_TYPE > a( N, M, P );
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
  constexpr INDEX_TYPE N = 10;

  {
    array< INDEX_TYPE > a( N );
    internal::testMemoryMotion( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotion( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionConst )
{
  constexpr INDEX_TYPE N = 100;

  {
    array< INDEX_TYPE > a( N );
    internal::testMemoryMotionConst( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionConst( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMove )
{
  constexpr INDEX_TYPE N = 100;

  {
    array< INDEX_TYPE > a( N );
    internal::testMemoryMotionMove( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMultiple )
{
  constexpr INDEX_TYPE N = 100;

  {
    array< INDEX_TYPE > a( N );
    internal::testMemoryMotionMultiple( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMultiple( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionMultipleMove )
{
  constexpr INDEX_TYPE N = 100;

  {
    array< INDEX_TYPE > a( N );
    internal::testMemoryMotionMultipleMove( a );
  }

  {
    array< Tensor > a( N );
    internal::testMemoryMotionMultipleMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray )
{
  constexpr INDEX_TYPE N = 10;

  {
    array< array< INDEX_TYPE > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArray( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArray( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArrayConst )
{
  constexpr INDEX_TYPE N = 10;

  {
    array< array< INDEX_TYPE > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayConst( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayConst( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArrayMove )
{
  constexpr INDEX_TYPE N = 10;

  {
    array< array< INDEX_TYPE > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( a );
  }

  {
    array< array< Tensor > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
    }

    internal::testMemoryMotionArrayMove( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray2 )
{
  constexpr INDEX_TYPE N = 5;

  {
    array< array< array< INDEX_TYPE > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( a );
  }

  {
    array< array< array< Tensor > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArray2Const )
{
  constexpr INDEX_TYPE N = 5;

  {
    array< array< array< INDEX_TYPE > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2Const( a );
  }

  {
    array< array< array< Tensor > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArray2Const( a );
  }
}

CUDA_TEST( ArrayView, memoryMotionArrayMove2 )
{
  constexpr INDEX_TYPE N = 5;

  {
    array< array< array< INDEX_TYPE > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArrayMove2( a );
  }

  {
    array< array< array< Tensor > > > a( N );
    for ( INDEX_TYPE i = 0; i < N; ++i )
    {
      a[ i ].resize( N );
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        a[ i ][ j ].resize( N );
      }
    }

    internal::testMemoryMotionArrayMove2( a );
  }
}

CUDA_TEST( ArrayView, 2DAccessorsDevice )
{
  constexpr INDEX_TYPE N = 20;
  constexpr INDEX_TYPE M = 10;

  {
    array2D< INDEX_TYPE > a( N, M );
    internal::test2DAccessorsDevice( a );
  }

  {
    array2D< Tensor > a( N, M );
    internal::test2DAccessorsDevice( a );
  }
}


CUDA_TEST( ArrayView, 3DAccessorsDevice )
{
  constexpr INDEX_TYPE N = 7;
  constexpr INDEX_TYPE M = 8;
  constexpr INDEX_TYPE P = 9;

  {
    array3D< INDEX_TYPE > a( N, M, P );
    internal::test3DAccessorsDevice( a );
  }

  {
    array3D< Tensor > a( N, M, P );
    internal::test3DAccessorsDevice( a );
  }
}

CUDA_TEST( ArrayView, sizeOnDevice )
{
  constexpr INDEX_TYPE N = 7;
  constexpr INDEX_TYPE M = 8;
  constexpr INDEX_TYPE P = 9;

  {
    array3D< INDEX_TYPE > a( N, M, P );
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
