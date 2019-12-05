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
#include "Array.hpp"

using namespace LvArray;

TEST( SizeHelper, SizeHelper )
{
  int dims[4] = { 3, 7, 11, 17 };

  int const result0 = multiplyAll< 4 >( dims );
  int const result1 = multiplyAll< 3 >( dims + 1 );
  int const result2 = multiplyAll< 2 >( dims + 2 );
  int const result3 = multiplyAll< 1 >( dims + 3 );
  ASSERT_EQ( result0, dims[0] * dims[1] * dims[2] * dims[3] );
  ASSERT_EQ( result1, dims[1] * dims[2] * dims[3] );
  ASSERT_EQ( result2, dims[2] * dims[3] );
  ASSERT_EQ( result3, dims[3] );
}

TEST( helpers, getLinearIndex )
{
  constexpr int NDIM = 4;
  constexpr int UNIT_STRIDE_DIM = 3;

  int const dims[ NDIM ] = { 3, 2, 5, 3 };
  int const strides[ NDIM ] = { dims[ 1 ] * dims[ 2 ] * dims[ 3 ],
                                dims[ 2 ] * dims[ 3 ],
                                dims[ 3 ],
                                -1000 }; // The stride of the UNIT_STRIDE_DIM shouldn't be used.

  for( int a0 = 0 ; a0 < dims[ 0 ] ; ++a0 )
  {
    for( int a1 = 0 ; a1 < dims[ 1 ] ; ++a1 )
    {
      for( int a2 = 0 ; a2 < dims[ 2 ] ; ++a2 )
      {
        for( int a3 = 0 ; a3 < dims[ 3 ] ; ++a3 )
        {
          int const expectedIndex = a0 * strides[ 0 ] + a1 * strides[ 1 ] + a2 * strides[ 2 ] + a3;
          int const calculatedIndex = getLinearIndex< UNIT_STRIDE_DIM >( strides, a0, a1, a2, a3 );
          EXPECT_EQ( expectedIndex, calculatedIndex ) << "a0 = " << a0 << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3;
        }
      }
    }
  }
}

TEST( helpers, getLinearIndexPermuted )
{
  constexpr int NDIM = 3;
  constexpr int UNIT_STRIDE_DIM = 0;

  // Permutation = { 2, 1, 0 }
  int const dims[ NDIM ] = { 3, 2, 5 };
  int const strides[ NDIM ] = { -1000, // The stride of the UNIT_STRIDE_DIM shouldn't be used.
                                dims[ 0 ],
                                dims[ 0 ] * dims[ 1 ] };

  for( int a0 = 0 ; a0 < dims[ 0 ] ; ++a0 )
  {
    for( int a1 = 0 ; a1 < dims[ 1 ] ; ++a1 )
    {
      for( int a2 = 0 ; a2 < dims[ 2 ] ; ++a2 )
      {
        int const expectedIndex = a0 + a1 * strides[ 1 ] + a2 * strides[ 2 ];
        int const calculatedIndex = getLinearIndex< UNIT_STRIDE_DIM >( strides, a0, a1, a2 );
        EXPECT_EQ( expectedIndex, calculatedIndex ) << "a0 = " << a0 << ", a1 = " << a1 << ", a2 = " << a2;
      }
    }
  }
}

TEST( helpers, getLinearIndexPermuted2 )
{
  constexpr int NDIM = 3;
  constexpr int UNIT_STRIDE_DIM = 0;

  // Permutation = { 1, 2, 0 }
  int const dims[ NDIM ] = { 3, 2, 5 };
  int const strides[ NDIM ] = { dims[ 1 ],
                                -1000, // The stride of the UNIT_STRIDE_DIM shouldn't be used.
                                dims[ 1 ] * dims[ 2 ] };

  for( int a0 = 0 ; a0 < dims[ 0 ] ; ++a0 )
  {
    for( int a1 = 0 ; a1 < dims[ 1 ] ; ++a1 )
    {
      for( int a2 = 0 ; a2 < dims[ 2 ] ; ++a2 )
      {
        int const expectedIndex = a0 + a1 * strides[ 1 ] + a2 * strides[ 2 ];
        int const calculatedIndex = getLinearIndex< UNIT_STRIDE_DIM >( strides, a0, a1, a2 );
        EXPECT_EQ( expectedIndex, calculatedIndex ) << "a0 = " << a0 << ", a1 = " << a1 << ", a2 = " << a2;
      }
    }
  }
}

TEST( helpers, checkIndices )
{
  int dims[4] = { 3, 2, 1, 3 };
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  EXPECT_TRUE( invalidIndices( dims, -1, 0, 0, 0 ) );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, -1, 0, 0, 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, -1, 0, 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, 0, -1, 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, 0, 0, -1 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, dims[0], 0, 0, 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, dims[1], 0, 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, 0, dims[2], 0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( checkIndices( dims, 0, 0, 0, dims[3] ), "" );

  for( int a0=0 ; a0<dims[0] ; ++a0 )
  {
    for( int a1=0 ; a1<dims[1] ; ++a1 )
    {
      for( int a2=0 ; a2<dims[2] ; ++a2 )
      {
        for( int a3=0 ; a3<dims[3] ; ++a3 )
        {
          ASSERT_NO_FATAL_FAILURE( checkIndices( dims, a0, a1, a2, a3 ) );
        }
      }
    }
  }
}
