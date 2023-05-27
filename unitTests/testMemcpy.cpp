/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "memcpy.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

template< template< typename > class BUFFER_TYPE >
void testMemcpy1D()
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );

  memcpy( y.toSlice(), x.toSliceConst() );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], y[ i ] );
  }

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = 2 * i;
  }

  memcpy< 0, 0 >( y, {}, x.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], y[ i ] );
  }
}

template< template< typename > class BUFFER_TYPE >
void testAsyncMemcpy1D()
{
  camp::resources::Resource host{ camp::resources::Host{} };

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );

  camp::resources::Event e = memcpy( host, y.toSlice(), x.toSliceConst() );
  host.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], y[ i ] );
  }

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = 2 * i;
  }

  e = memcpy< 0, 0 >( host, y, {}, x.toViewConst(), {} );
  host.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], y[ i ] );
  }
}

template< template< typename > class BUFFER_TYPE >
void testMemcpy2D()
{
  Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, BUFFER_TYPE > x( 2, 10 );

  for( std::ptrdiff_t i = 0; i < x.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < x.size( 1 ); ++j )
    {
      x( i, j ) = 4 * i + j;
    }
  }

  Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, BUFFER_TYPE > y( x.size( 0 ), x.size( 1 ) );

  memcpy( y[ 0 ], x[ 1 ].toSliceConst() );
  memcpy( y[ 1 ], x[ 0 ].toSliceConst() );

  for( std::ptrdiff_t i = 0; i < x.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < x.size( 1 ); ++j )
    {
      EXPECT_EQ( x( i, j ), y( 1 - i, j ) );
    }
  }

  memcpy< 1, 1 >( y, { 1 }, x.toViewConst(), { 1 } );

  for( std::ptrdiff_t i = 0; i < x.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < x.size( 1 ); ++j )
    {
      EXPECT_EQ( x( 1, j ), y( i, j ) );
    }
  }

  memcpy< 0, 0 >( y, {}, x.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < x.size( 1 ); ++j )
    {
      EXPECT_EQ( x( i, j ), y( i, j ) );
    }
  }
}

template< template< typename > class BUFFER_TYPE >
void testInvalidMemcpy()
{
  Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, BUFFER_TYPE > x( 6, 7 );
  Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, BUFFER_TYPE > y( 4, 3 );

  EXPECT_DEATH_IF_SUPPORTED( memcpy( x.toSlice(), y.toSliceConst() ), "" );
}

#if defined( LVARRAY_USE_CUDA )

template< template< typename > class BUFFER_TYPE >
void testMemcpyDevice()
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );
  y.move( MemorySpace::cuda );
  int * yPtr = y.data();

  memcpy< 0, 0 >( y, {}, x.toViewConst(), {} );

  forall< RAJA::cuda_exec< 32 > >( y.size(), [yPtr] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        PORTABLE_EXPECT_EQ( yPtr[ i ], i );
        yPtr[ i ] *= 2;
      } );

  memcpy< 0, 0 >( x, {}, y.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], 2 * i );
  }

  // Move y to the CPU but then capture and modify a view on device. This way y's data pointer is still pointing
  // to host memory but the subsequent memcpy should pick up that it's previous space is on device.
  y.move( MemorySpace::host );

  ArrayView< int, 1, 0, std::ptrdiff_t, BUFFER_TYPE > const yView = y.toView();
  forall< RAJA::cuda_exec< 32 > >( y.size(), [yView] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        yView[ i ] = -i;
      } );

  memcpy< 0, 0 >( x, {}, y.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], -i );
  }
}

template< template< typename > class BUFFER_TYPE >
void testAsyncMemcpyDevice()
{
  camp::resources::Resource stream{ camp::resources::Cuda{} };

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );
  y.move( MemorySpace::cuda );
  int * yPtr = y.data();

  camp::resources::Event e = memcpy< 0, 0 >( stream, y.toView(), {}, x.toViewConst(), {} );
  stream.wait_for( &e );

  forall< RAJA::cuda_exec< 32 > >( y.size(), [yPtr] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        PORTABLE_EXPECT_EQ( yPtr[ i ], i );
        yPtr[ i ] *= 2;
      } );

  e = memcpy< 0, 0 >( stream, x, {}, y.toViewConst(), {} );
  stream.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], 2 * i );
  }

  // Move y to the CPU but then capture and modify a view on device. This way y's data pointer is still pointing
  // to host memory but the subsequent memcpy should pick up that it's previous space is on device.
  y.move( MemorySpace::host );

  ArrayView< int, 1, 0, std::ptrdiff_t, BUFFER_TYPE > const yView = y.toView();
  forall< RAJA::cuda_exec< 32 > >( y.size(), [yView] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        yView[ i ] = -i;
      } );

  e = memcpy< 0, 0 >( stream, x, {}, y.toViewConst(), {} );
  stream.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], -i );
  }
}
#elif defined(LVARRAY_USE_HIP)

template< template< typename > class BUFFER_TYPE >
void testMemcpyDevice()
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );
  y.move( MemorySpace::hip );
  int * yPtr = y.data();

  memcpy< 0, 0 >( y, {}, x.toViewConst(), {} );

  forall< RAJA::hip_exec< 32 > >( y.size(), [yPtr] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        PORTABLE_EXPECT_EQ( yPtr[ i ], i );
        yPtr[ i ] *= 2;
      } );

  memcpy< 0, 0 >( x, {}, y.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], 2 * i );
  }

  // Move y to the CPU but then capture and modify a view on device. This way y's data pointer is still pointing
  // to host memory but the subsequent memcpy should pick up that it's previous space is on device.
  y.move( MemorySpace::host );

  ArrayView< int, 1, 0, std::ptrdiff_t, BUFFER_TYPE > const yView = y.toView();
  forall< RAJA::hip_exec< 32 > >( y.size(), [yView] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        yView[ i ] = -i;
      } );

  memcpy< 0, 0 >( x, {}, y.toViewConst(), {} );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], -i );
  }
}

template< template< typename > class BUFFER_TYPE >
void testAsyncMemcpyDevice()
{
  camp::resources::Resource stream{ camp::resources::Hip{} };

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > x( 100 );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    x[ i ] = i;
  }

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > y( x.size() );
  y.move( MemorySpace::hip );
  int * yPtr = y.data();

  camp::resources::Event e = memcpy< 0, 0 >( stream, y.toView(), {}, x.toViewConst(), {} );
  stream.wait_for( &e );

  forall< RAJA::hip_exec< 32 > >( y.size(), [yPtr] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        PORTABLE_EXPECT_EQ( yPtr[ i ], i );
        yPtr[ i ] *= 2;
      } );

  e = memcpy< 0, 0 >( stream, x, {}, y.toViewConst(), {} );
  stream.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], 2 * i );
  }

  // Move y to the CPU but then capture and modify a view on device. This way y's data pointer is still pointing
  // to host memory but the subsequent memcpy should pick up that it's previous space is on device.
  y.move( MemorySpace::host );

  ArrayView< int, 1, 0, std::ptrdiff_t, BUFFER_TYPE > const yView = y.toView();
  forall< RAJA::hip_exec< 32 > >( y.size(), [yView] LVARRAY_DEVICE ( std::ptrdiff_t const i )
      {
        yView[ i ] = -i;
      } );

  e = memcpy< 0, 0 >( stream, x, {}, y.toViewConst(), {} );
  stream.wait_for( &e );

  for( std::ptrdiff_t i = 0; i < x.size(); ++i )
  {
    EXPECT_EQ( x[ i ], -i );
  }
}
#endif

TEST( TestMemcpy, MallocBuffer1D )
{
  testMemcpy1D< MallocBuffer >();
}

TEST( TestMemcpy, AsyncMallocBuffer1D )
{
  testAsyncMemcpy1D< MallocBuffer >();
}

TEST( TestMemcpy, MallocBuffer2D )
{
  testMemcpy2D< MallocBuffer >();
}

TEST( TestMemcpy, Invalid )
{
  testInvalidMemcpy< MallocBuffer >();
}

#if defined(LVARRAY_USE_CHAI)

TEST( TestMemcpy, ChaiBuffer1D )
{
  testMemcpy1D< ChaiBuffer >();
}

TEST( TestMemcpy, AsyncChaiBuffer1D )
{
  testAsyncMemcpy1D< ChaiBuffer >();
}

TEST( TestMemcpy, ChaiBuffer2D )
{
  testMemcpy2D< ChaiBuffer >();
}

#if defined( LVARRAY_USE_CUDA ) || defined( LVARRAY_USE_HIP )

TEST( TestMemcpy, ChaiBufferDevice )
{
  testMemcpyDevice< ChaiBuffer >();
}

TEST( TestMemcpy, AsyncChaiBufferDevice )
{
  testAsyncMemcpyDevice< ChaiBuffer >();
}

#endif

#endif

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
