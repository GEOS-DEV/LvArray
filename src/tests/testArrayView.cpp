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

// Source includes
#include "Array.hpp"
#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

#if defined( USE_CUDA ) && defined( USE_CHAI )
  #include <chai/util/forall.hpp>
#endif

// System includes
#include <type_traits>
#include <random>

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

int rand( int const low, int const high )
{
  static std::mt19937_64 m_gen;
  return std::uniform_int_distribution< int >( low, high )( m_gen );
}

template< int UNIT_STRIDE_DIM, typename LAYOUT >
void compareArrayToRAJAView( LvArray::ArrayView< int, 2, UNIT_STRIDE_DIM > const & v,
                             RAJA::View< int, LAYOUT > view )
{
  ASSERT_EQ( v.size( 0 ), view.layout.sizes[ 0 ] );
  ASSERT_EQ( v.size( 1 ), view.layout.sizes[ 1 ] );
  
  int const * const data = v.data();
  ASSERT_EQ( data, view.data );

  for ( int i = 0; i < v.size( 0 ); ++i )
  {
    for ( int j = 0; j < v.size( 1 ); ++j )
    {
      EXPECT_EQ( &view( i, j ) - data, &v[ i ][ j ] - data );
      EXPECT_EQ( &view( i, j ) - data, &v( i, j ) - data );
    }
  }
}

template< int UNIT_STRIDE_DIM, typename LAYOUT >
void compareArrayToRAJAView( LvArray::ArrayView< int, 3, UNIT_STRIDE_DIM > const & v,
                             RAJA::View< int, LAYOUT > view )
{
  ASSERT_EQ( v.size( 0 ), view.layout.sizes[ 0 ] );
  ASSERT_EQ( v.size( 1 ), view.layout.sizes[ 1 ] );
  ASSERT_EQ( v.size( 2 ), view.layout.sizes[ 2 ] );

  int const * const data = v.data();
  ASSERT_EQ( data, view.data );

  for ( int i = 0; i < v.size( 0 ); ++i )
  {
    for ( int j = 0; j < v.size( 1 ); ++j )
    {
      for ( int k = 0; k < v.size( 2 ); ++k )
      {
        EXPECT_EQ( &view( i, j, k ) - data, &v[ i ][ j ][ k ] - data );
        EXPECT_EQ( &view( i, j, k ) - data, &v( i, j, k ) - data ) << i << ", " << j << ", " << k;
      }
    }
  }
}

template< int UNIT_STRIDE_DIM, typename LAYOUT >
void compareArrayToRAJAView( LvArray::ArrayView< int, 4, UNIT_STRIDE_DIM > const & v,
                             RAJA::View< int, LAYOUT > view )
{
  ASSERT_EQ( v.size( 0 ), view.layout.sizes[ 0 ] );
  ASSERT_EQ( v.size( 1 ), view.layout.sizes[ 1 ] );
  ASSERT_EQ( v.size( 2 ), view.layout.sizes[ 2 ] );
  ASSERT_EQ( v.size( 3 ), view.layout.sizes[ 3 ] );

  int const * const data = v.data();
  ASSERT_EQ( data, view.data );

  for ( int i = 0; i < v.size( 0 ); ++i )
  {
    for ( int j = 0; j < v.size( 1 ); ++j )
    {
      for ( int k = 0; k < v.size( 2 ); ++k )
      {
        for ( int l = 0; l < v.size( 3 ); ++l )
        {
          EXPECT_EQ( &view( i, j, k, l ) - data, &v[ i ][ j ][ k ][ l ] - data );
          EXPECT_EQ( &view( i, j, k, l ) - data, &v( i, j, k, l ) - data );
          EXPECT_EQ( view( i, j, k, l ), v( i, j, k, l ) );
        }
      }
    }
  }
}

template< typename PERMUTATION >
void testArrayPermutation()
{
  constexpr int NDIM = LvArray::getDimension( PERMUTATION {} );
  LvArray::Array< int, NDIM, PERMUTATION > a;

  std::array< INDEX_TYPE, NDIM > dimensions;
  for ( int i = 0; i < NDIM; ++i )
  {
    dimensions[ i ] = rand( 1, 10 );
  }

  a.resize( NDIM, dimensions.data() );

  RAJA::View< int, RAJA::Layout< NDIM > > view( a.data(), RAJA::make_permuted_layout( dimensions, RAJA::as_array< PERMUTATION >::get() ) );

  for ( int i = 0; i < NDIM; ++i )
  {
    EXPECT_EQ( a.strides()[ i ], view.layout.strides[ i ] );
  }

  for ( int i = 0; i < a.size(); ++i )
  {
    a.data()[ i ] = i;
  }

  compareArrayToRAJAView( a, view );
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += v[ i ];
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      T val( i );
      val += val;
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
  forall( gpu(), 0, N,
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += v[ i ];
    }
  );

  /* The copy construction shouldn't have changed the pointer since we moved it. */
  ASSERT_EQ(v_device_ptr, v.data());

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    T val = T( i );
    val += val;
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );


  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      T val( i );
      val += T( 2 );
      val += T( 2 );
      val += T( 2 );
      val += T( 2 );
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    v[ i ] += T( 2 );
  }

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      v[ i ] += T( 2 );
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    T val( i );
    val += T( 2 );
    val += T( 2 );
    val += T( 2 );
    val += T( 2 );
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        v[ i ][ j ] += v[ i ][ j ];
      }
    }
  );

  forall( sequential(), 0, N,
    [=]( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        T val( N * i + j );
        val += val;
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
  array< arrayView_nc< T > > a_copy;
  a_copy.resize(N);

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
  forall( gpu(), 0, N,
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        v[ i ][ j ] += v[ i ][ j ];
      }
    }
  );

  a.move( chai::CPU );
  for ( INDEX_TYPE i = 0; i < N; ++i )
  {
    for ( INDEX_TYPE j = 0; j < N; ++j )
    {
      T val( N * i + j );
      val += val;
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] += v[ i ][ j ][ k ];
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
          val += val;
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

  forall( gpu(), 0, N,
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

  forall( gpu(), 0, N,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < N; ++j )
      {
        for ( INDEX_TYPE k = 0; k < N; ++k )
        {
          v[ i ][ j ][ k ] += v[ i ][ j ][ k ];
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
        val += val;
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

  forall( gpu(), 0, I,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        v( i, j ) += T( 2 );
        v[ i ][ j ] += T( 2 );
      }
    }
  );

  forall( sequential(), 0, I,
    [=] ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        T val( J * i + j  );
        val += T( 2 );
        val += T( 2 );
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

  forall( gpu(), 0, I,
    [=] __device__ ( INDEX_TYPE i )
    {
      for ( INDEX_TYPE j = 0; j < J; ++j )
      {
        for ( INDEX_TYPE k = 0; k < K; ++k )
        {
          v( i, j, k ) += T( 2 );
          v[ i ][ j ][ k ] += T( 2 );
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
          val += T( 2 );
          val += T( 2 );
          EXPECT_EQ( v[ i ][ j ][ k ], val );
        }
      }
    }
  );
}

template < typename T >
void testSizeOnDevice( arrayView3D< T > & v )
{

  forall( gpu(), 0, v.size(0),
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


TEST( ArrayView, Permutations2D )
{
  internal::testArrayPermutation< RAJA::PERM_IJ >();
  internal::testArrayPermutation< RAJA::PERM_JI >();
}

TEST( ArrayView, Permutations3D )
{
  internal::testArrayPermutation< RAJA::PERM_IJK >();
  internal::testArrayPermutation< RAJA::PERM_JIK >();
  internal::testArrayPermutation< RAJA::PERM_IKJ >();
  internal::testArrayPermutation< RAJA::PERM_KIJ >();
  internal::testArrayPermutation< RAJA::PERM_JKI >();
  internal::testArrayPermutation< RAJA::PERM_KJI >();
}

TEST( ArrayView, Permutations4D )
{
  internal::testArrayPermutation< RAJA::PERM_IJKL >();
  internal::testArrayPermutation< RAJA::PERM_JIKL >();
  internal::testArrayPermutation< RAJA::PERM_IKJL >();
  internal::testArrayPermutation< RAJA::PERM_KIJL >();
  internal::testArrayPermutation< RAJA::PERM_JKIL >();
  internal::testArrayPermutation< RAJA::PERM_KJIL >();
  internal::testArrayPermutation< RAJA::PERM_IJLK >();
  internal::testArrayPermutation< RAJA::PERM_JILK >();
  internal::testArrayPermutation< RAJA::PERM_ILJK >();
  internal::testArrayPermutation< RAJA::PERM_LIJK >();
  internal::testArrayPermutation< RAJA::PERM_JLIK >();
  internal::testArrayPermutation< RAJA::PERM_LJIK >();
  internal::testArrayPermutation< RAJA::PERM_IKLJ >();
  internal::testArrayPermutation< RAJA::PERM_KILJ >();
  internal::testArrayPermutation< RAJA::PERM_ILKJ >();
  internal::testArrayPermutation< RAJA::PERM_LIKJ >();
  internal::testArrayPermutation< RAJA::PERM_KLIJ >();
  internal::testArrayPermutation< RAJA::PERM_LKIJ >();
  internal::testArrayPermutation< RAJA::PERM_JKLI >();
  internal::testArrayPermutation< RAJA::PERM_KJLI >();
  internal::testArrayPermutation< RAJA::PERM_JLKI >();
  internal::testArrayPermutation< RAJA::PERM_LJKI >();
  internal::testArrayPermutation< RAJA::PERM_KLJI >();
  internal::testArrayPermutation< RAJA::PERM_LKJI >();
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
