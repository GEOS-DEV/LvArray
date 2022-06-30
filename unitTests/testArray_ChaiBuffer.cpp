/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file testBuffers.cpp
 */

// Source includes
#include "testUtils.hpp"
#include "ChaiBuffer.hpp"
#include "Array.hpp"
#include "umpire/strategy/QuickPool.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <vector>
#include <random>


namespace LvArray
{
namespace testing
{

template< typename T >
class ArrayTest : public ::testing::Test
{
public:

  void testAllocatorConstruction()
  {
    umpire::ResourceManager & rm = umpire::ResourceManager::getInstance();
    auto hostPool = rm.makeAllocator< umpire::strategy::QuickPool >( "HOST_pool", rm.getAllocator( "HOST" ) );

  #if defined( LVARRAY_USE_CUDA )
    auto devicePool = rm.makeAllocator< umpire::strategy::QuickPool >( "DEVICE_pool", rm.getAllocator( "DEVICE" ) );
    std::initializer_list< MemorySpace > const spaces = { MemorySpace::host, MemorySpace::cuda };
    std::initializer_list< umpire::Allocator > const allocators = { hostPool, devicePool };
  #elif defined(LVARRAY_USE_HIP)
    auto devicePool = rm.makeAllocator< umpire::strategy::QuickPool >( "DEVICE_pool", rm.getAllocator( "DEVICE" ) );
    std::initializer_list< MemorySpace > const spaces = { MemorySpace::host, MemorySpace::hip };
    std::initializer_list< umpire::Allocator > const allocators = { hostPool, devicePool };
  #else
    std::initializer_list< MemorySpace > const spaces = { MemorySpace::host };
    std::initializer_list< umpire::Allocator > const allocators = { hostPool };
  #endif

    Array< int, 1, RAJA::PERM_I, int, ChaiBuffer > array( ChaiBuffer< T >( spaces, allocators ) );
    array.resize( 100 );

    for( int i = 0; i < array.size(); ++i )
    {
      array[ i ] = T( i );
    }

    EXPECT_EQ( rm.getAllocator( array.data() ).getName(), "HOST_pool" );

  #if defined( LVARRAY_USE_CUDA )
    array.move( MemorySpace::cuda, true );
    EXPECT_EQ( rm.getAllocator( array.data() ).getName(), "DEVICE_pool" );

    array.move( MemorySpace::host, true );
    EXPECT_EQ( rm.getAllocator( array.data() ).getName(), "HOST_pool" );
  #elif defined(LVARRAY_USE_HIP)
    array.move( MemorySpace::hip, true );
    EXPECT_EQ( rm.getAllocator( array.data() ).getName(), "DEVICE_pool" );

    array.move( MemorySpace::host, true );
    EXPECT_EQ( rm.getAllocator( array.data() ).getName(), "HOST_pool" );
  #endif
  }

#if defined( LVARRAY_USE_CUDA )
  void testCudaDeviceAlloc()
  {
    Array< int, 1, RAJA::PERM_I, int, ChaiBuffer > array;

    array.resizeWithoutInitializationOrDestruction( MemorySpace::cuda, 100 );

    T * const devPtr = array.data();
    forall< parallelDevicePolicy< 32 > >( array.size(), [devPtr] LVARRAY_DEVICE ( int const i )
        {
          new ( &devPtr[ i ] ) T( i );
        } );

    array.move( MemorySpace::host, true );
    for( int i = 0; i < array.size(); ++i )
    {
      EXPECT_EQ( array[ i ], T( i ) );
    }
  }
#endif
#if defined(LVARRAY_USE_HIP)
  void testHIPDeviceAlloc()
  {
    Array< int, 1, RAJA::PERM_I, int, ChaiBuffer > array;

    array.resizeWithoutInitializationOrDestruction( MemorySpace::hip, 100 );

    T * const devPtr = array.data();
    forall< parallelDevicePolicy< 32 > >( array.size(), [devPtr] LVARRAY_DEVICE ( int const i )
        {
          new ( &devPtr[ i ] ) T( i );
        } );

    array.move( MemorySpace::host, true );
    for( int i = 0; i < array.size(); ++i )
    {
      EXPECT_EQ( array[ i ], T( i ) );
    }
  }
#endif
};

/// The list of types to instantiate ArrayTest with.
using ArrayTestTypes = ::testing::Types<
  int
  >;

TYPED_TEST_SUITE( ArrayTest, ArrayTestTypes, );

TYPED_TEST( ArrayTest, AllocatorConstruction )
{
  this->testAllocatorConstruction();
}

#if defined( LVARRAY_USE_CUDA )

TYPED_TEST( ArrayTest, DeviceAlloc )
{
  this->testCudaDeviceAlloc();
}

#endif
#if defined(LVARRAY_USE_HIP)

TYPED_TEST( ArrayTest, DeviceAlloc )
{
  this->testHIPDeviceAlloc();
}

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
