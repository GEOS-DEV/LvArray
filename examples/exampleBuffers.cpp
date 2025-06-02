/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "MallocBuffer.hpp"
#include "StackBuffer.hpp"
#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// system includes
#include <string>

// You can't define a nvcc extended host-device or device lambda in a TEST statement.
#define CUDA_TEST( X, Y )                 \
  static void cuda_test_ ## X ## _ ## Y();    \
  TEST( X, Y ) { cuda_test_ ## X ## _ ## Y(); } \
  static void cuda_test_ ## X ## _ ## Y()

// Sphinx start after MallocBuffer
TEST( MallocBuffer, copy )
{
  constexpr std::ptrdiff_t size = 55;
  LvArray::MallocBuffer< int > buffer( true );
  buffer.reallocate( 0, LvArray::MemorySpace::host, size );

  for( int i = 0; i < size; ++i )
  {
    buffer[ i ] = i;
  }

  for( int i = 0; i < size; ++i )
  {
    EXPECT_EQ( buffer[ i ], i );
  }

  // MallocBuffer has shallow copy semantics.
  LvArray::MallocBuffer< int > copy = buffer;
  EXPECT_EQ( copy.data(), buffer.data() );

  // Must be manually free'd.
  buffer.free();
}

TEST( MallocBuffer, nonPOD )
{
  constexpr std::ptrdiff_t size = 4;
  LvArray::MallocBuffer< std::string > buffer( true );
  buffer.reallocate( 0, LvArray::MemorySpace::host, size );

  // Buffers don't initialize data so placement new must be used.
  for( int i = 0; i < size; ++i )
  {
    new ( buffer.data() + i ) std::string( std::to_string( i ) );
  }

  for( int i = 0; i < size; ++i )
  {
    EXPECT_EQ( buffer[ i ], std::to_string( i ) );
  }

  // Buffers don't destroy the objects in free.
  // The using statement is needed to explicitly call the destructor
  using std::string;
  for( int i = 0; i < size; ++i )
  {
    buffer[ i ].~string();
  }

  buffer.free();
}
// Sphinx end before MallocBuffer

#if defined(LVARRAY_USE_CHAI) && defined(LVARRAY_USE_CUDA)
// Sphinx start after ChaiBuffer captureOnDevice
CUDA_TEST( ChaiBuffer, captureOnDevice )
{
  constexpr std::ptrdiff_t size = 55;
  LvArray::ChaiBuffer< int > buffer( true );
  buffer.reallocate( 0, LvArray::MemorySpace::host, size );

  for( int i = 0; i < size; ++i )
  {
    buffer[ i ] = i;
  }

  // Capture buffer in a device kernel which creates an allocation on device
  // and copies the data there.
  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, size ),
    [buffer] __device__ ( std::ptrdiff_t const i )
  {
    buffer[ i ] += i;
  } );

  // Capture buffer in a host kernel moving the data back to the host allocation.
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, size ),
    [buffer] ( std::ptrdiff_t const i )
  {
    EXPECT_EQ( buffer[ i ], 2 * i );
  } );

  buffer.free();
}
// Sphinx end before ChaiBuffer captureOnDevice

// Sphinx start after ChaiBuffer captureOnDeviceConst
CUDA_TEST( ChaiBuffer, captureOnDeviceConst )
{
  constexpr std::ptrdiff_t size = 55;
  LvArray::ChaiBuffer< int > buffer( true );
  buffer.reallocate( 0, LvArray::MemorySpace::host, size );

  for( int i = 0; i < size; ++i )
  {
    buffer[ i ] = i;
  }

  // Create a const buffer and capture it in a device kernel which
  // creates an allocation on device and copies the data there.
  LvArray::ChaiBuffer< int const > const constBuffer( buffer );
  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, size ),
    [constBuffer] __device__ ( std::ptrdiff_t const i )
  {
    const_cast< int & >( constBuffer[ i ] ) += i;
  } );

  // Capture buffer in a host kernel moving the data back to the host allocation.
  // If constBuffer didn't contain "int const" then this check would fail because
  // the data would be copied back from device.
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, size ),
    [buffer] ( std::ptrdiff_t const i )
  {
    EXPECT_EQ( buffer[ i ], i );
  } );

  buffer.free();
}
// Sphinx end before ChaiBuffer captureOnDeviceConst

// Sphinx start after ChaiBuffer setName
TEST( ChaiBuffer, setName )
{
  LvArray::ChaiBuffer< int > buffer( true );
  buffer.reallocate( 0, LvArray::MemorySpace::host, 1024 );

  // Move to the device.
  buffer.move( LvArray::MemorySpace::cuda, true );

  // Give buffer a name and move back to the host.
  buffer.setName( "my_buffer" );
  buffer.move( LvArray::MemorySpace::host, true );

  // Rename buffer and override the default type.
  buffer.setName< double >( "my_buffer_with_a_nonsensical_type" );
  buffer.move( LvArray::MemorySpace::cuda, true );
}
// Sphinx end before ChaiBuffer setName

#endif

// Sphinx start after ChaiBuffer StackBuffer
TEST( StackBuffer, example )
{
  constexpr std::ptrdiff_t size = 55;
  LvArray::StackBuffer< int, 55 > buffer( true );

  static_assert( buffer.capacity() == size, "Capacity is fixed at compile time." );

  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    buffer[ i ] = i;
  }

  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    EXPECT_EQ( buffer[ i ], i );
  }

  EXPECT_DEATH_IF_SUPPORTED( buffer.reallocate( size, LvArray::MemorySpace::host, 2 * size ), "" );

  // Not necessary with the StackBuffer but it's good practice.
  buffer.free();
}
// Sphinx end before ChaiBuffer StackBuffer

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
