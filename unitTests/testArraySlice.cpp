/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{

template< typename T, typename PERMUTATION >
using ArrayT = Array< T, typeManipulation::getDimension< PERMUTATION >, PERMUTATION, std::ptrdiff_t, MallocBuffer >;

template< typename ARRAY >
class ArraySliceTest : public ::testing::Test
{
public:
  static constexpr int NDIM = ARRAY::NDIM;
  using PERMUTATION = typename ARRAY::Permutation;

  void resize()
  {
    // Create a permuted array and resize with nontrivial sizes
    std::array< int, NDIM > dims{};
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 2 + i; }
    m_array.resize( NDIM, dims.data() );
  }

  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 2 >
  isContiguous()
  { checkOneSlice(); }

  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 3 >
  isContiguous()
  {
    checkOneSlice();
    checkTwoSlices();
  }

  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 4 >
  isContiguous()
  {
    checkOneSlice();
    checkTwoSlices();
    checkThreeSlices();
  }

protected:

  void checkOneSlice()
  {
    // The only way to get a contiguous (NDIM-1)-dimensional slice is by
    // chopping along the largest-stride dimension (i.e. first in permutation)
    bool const contiguous = RAJA::as_array< PERMUTATION >::get()[ 0 ] == 0;

    // There's no need to test all slices, but we might as well
    for( int i = 0; i < m_array.size( 0 ); ++i )
    { EXPECT_EQ( m_array[ i ].isContiguous(), contiguous ); }
  }

  void checkTwoSlices()
  {
    bool const contiguous = RAJA::as_array< PERMUTATION >::get()[ 0 ] <= 1 &&
                            RAJA::as_array< PERMUTATION >::get()[ 1 ] <= 1;

    for( int i = 0; i < m_array.size( 0 ); ++i )
    {
      for( int j = 0; j < m_array.size( 1 ); ++j )
      {
        EXPECT_EQ( m_array[ i ][ j ].isContiguous(), contiguous );
      }
    }
  }

  void checkThreeSlices()
  {
    bool const contiguous = RAJA::as_array< PERMUTATION >::get()[ 0 ] <= 2 &&
                            RAJA::as_array< PERMUTATION >::get()[ 1 ] <= 2 &&
                            RAJA::as_array< PERMUTATION >::get()[ 2 ] <= 2;

    for( int i = 0; i < m_array.size( 0 ); ++i )
    {
      for( int j = 0; j < m_array.size( 1 ); ++j )
      {
        for( int k = 0; k < m_array.size( 2 ); ++k )
        {
          EXPECT_EQ( m_array[ i ][ j ][ k ].isContiguous(), contiguous );
        }
      }
    }
  }

  ARRAY m_array;
};

using ArraySliceTestTypes = ::testing::Types<
  ArrayT< int, RAJA::PERM_IJ >
  , ArrayT< int, RAJA::PERM_JI >
  , ArrayT< int, RAJA::PERM_IJK >
  , ArrayT< int, RAJA::PERM_JIK >
  , ArrayT< int, RAJA::PERM_IKJ >
  , ArrayT< int, RAJA::PERM_KIJ >
  , ArrayT< int, RAJA::PERM_KJI >
  , ArrayT< int, RAJA::PERM_JKI >
  , ArrayT< int, RAJA::PERM_IJKL >
  , ArrayT< int, RAJA::PERM_JIKL >
  , ArrayT< int, RAJA::PERM_IKJL >
  , ArrayT< int, RAJA::PERM_KIJL >
  , ArrayT< int, RAJA::PERM_JKIL >
  , ArrayT< int, RAJA::PERM_KJIL >
  , ArrayT< int, RAJA::PERM_IJLK >
  , ArrayT< int, RAJA::PERM_JILK >
  , ArrayT< int, RAJA::PERM_ILJK >
  , ArrayT< int, RAJA::PERM_LIJK >
  , ArrayT< int, RAJA::PERM_JLIK >
  , ArrayT< int, RAJA::PERM_LJIK >
  , ArrayT< int, RAJA::PERM_IKLJ >
  , ArrayT< int, RAJA::PERM_KILJ >
  , ArrayT< int, RAJA::PERM_ILKJ >
  , ArrayT< int, RAJA::PERM_LIKJ >
  , ArrayT< int, RAJA::PERM_KLIJ >
  , ArrayT< int, RAJA::PERM_LKIJ >
  , ArrayT< int, RAJA::PERM_JKLI >
  , ArrayT< int, RAJA::PERM_KJLI >
  , ArrayT< int, RAJA::PERM_JLKI >
  , ArrayT< int, RAJA::PERM_LJKI >
  , ArrayT< int, RAJA::PERM_KLJI >
  , ArrayT< int, RAJA::PERM_LKJI >
  >;

TYPED_TEST_SUITE( ArraySliceTest, ArraySliceTestTypes, );

TYPED_TEST( ArraySliceTest, IsContiguous )
{
  this->resize();
  this->checkOneSlice();
}

} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
