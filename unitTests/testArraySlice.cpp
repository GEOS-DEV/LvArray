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
#include "Array.hpp"

// TPL includes
#include <gtest/gtest.h>

using namespace LvArray;

template< typename T, int NDIM, typename PERM, typename INDEX_TYPE, typename DIMS, camp::idx_t ... INDICES >
void resizeArray( Array< T, NDIM, PERM, INDEX_TYPE > & a, DIMS const & newdims,
                  camp::idx_seq< INDICES... > )
{
  static_assert( sizeof...(INDICES) == NDIM, "Number of indices must be equal to number of dimensions" );
  a.resize( newdims[INDICES] ... );
}

template< typename PERMUTATION >
class PermutedSliceTest : public ::testing::Test {};

using Permutations = ::testing::Types< RAJA::PERM_IJ,
                                       RAJA::PERM_JI,
                                       RAJA::PERM_IJK,
                                       RAJA::PERM_JIK,
                                       RAJA::PERM_IKJ,
                                       RAJA::PERM_KIJ,
                                       RAJA::PERM_KJI,
                                       RAJA::PERM_JKI,
                                       RAJA::PERM_IJKL,
                                       RAJA::PERM_JIKL,
                                       RAJA::PERM_IKJL,
                                       RAJA::PERM_KIJL,
                                       RAJA::PERM_JKIL,
                                       RAJA::PERM_KJIL,
                                       RAJA::PERM_IJLK,
                                       RAJA::PERM_JILK,
                                       RAJA::PERM_ILJK,
                                       RAJA::PERM_LIJK,
                                       RAJA::PERM_JLIK,
                                       RAJA::PERM_LJIK,
                                       RAJA::PERM_IKLJ,
                                       RAJA::PERM_KILJ,
                                       RAJA::PERM_ILKJ,
                                       RAJA::PERM_LIKJ,
                                       RAJA::PERM_KLIJ,
                                       RAJA::PERM_LKIJ,
                                       RAJA::PERM_JKLI,
                                       RAJA::PERM_KJLI,
                                       RAJA::PERM_JLKI,
                                       RAJA::PERM_LJKI,
                                       RAJA::PERM_KLJI,
                                       RAJA::PERM_LKJI >;

TYPED_TEST_CASE( PermutedSliceTest, Permutations );

TYPED_TEST( PermutedSliceTest, IsContiguous )
{
  using PERMUTATION = TypeParam;
  constexpr int NDIM = getDimension( PERMUTATION {} );

  // Create a permuted array and resize with nontrivial sizes
  Array< int, NDIM, PERMUTATION, int > a;
  std::array< int, NDIM > dims{};
  for( int i = 0; i < NDIM; ++i )
  {
    dims[i] = 2+i;
  }
  resizeArray( a, dims, camp::make_idx_seq_t< NDIM >{} );

  // The only way to get a contiguous (NDIM-1)-dimenstional slice is by
  // chopping along the largest-stride dimension (i.e. first in permutation)
  bool const contiguous = RAJA::as_array< PERMUTATION >::get()[0] == 0;

  // There's no need to test all slices, but we might as well
  for( int i = 0; i < dims[0]; ++i )
  {
    EXPECT_EQ( a[i].isContiguous(), contiguous );
  }
}

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
