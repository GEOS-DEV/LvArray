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

#if defined(__APPLE__) &&  defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#endif

TEST( SizeHelper, SizeHelper )
{
  int dims[4] = { 3, 7, 11, 17 };

  int const result0 = size_helper<4,int,0>::f(dims);
  int const result1 = size_helper<4,int,1>::f(dims);
  int const result2 = size_helper<4,int,2>::f(dims);
  int const result3 = size_helper<4,int,3>::f(dims);
  ASSERT_TRUE( result0 == dims[0]*dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result1 ==         dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result2 ==                 dims[2]*dims[3] );
  ASSERT_TRUE( result3 ==                         dims[3] );

}


template< int NDIM, typename... INDICES >
int indexCaller( int const * const strides, INDICES... indices )
{
  return linearIndex_helper< NDIM, int, INDICES...>::evaluate(strides, indices...);
}

TEST( IndexHelper, IndexHelper )
{
  int dims[4] = { 3, 2, 5, 3 };
  int strides[4] = { dims[1]*dims[2]*dims[3],
                             dims[2]*dims[3],
                                     dims[3],
                                           1 };

  for( int a0=0 ; a0<dims[0] ; ++a0 )
  {
    for( int a1=0 ; a1<dims[1] ; ++a1 )
    {
      for( int a2=0 ; a2<dims[2] ; ++a2 )
      {
        for( int a3=0 ; a3<dims[3] ; ++a3 )
        {
          ASSERT_TRUE( indexCaller<4>( strides, a0,a1,a2,a3 ) ==
              a0*strides[0]+a1*strides[1]+a2*strides[2]+a3*strides[3] );
        }
      }
    }
  }
}


template< int NDIM, typename... INDICES >
void indexChecker( int const * const dims, INDICES... indices )
{
  linearIndex_helper< NDIM, int, INDICES...>::check(dims, indices...);
}

TEST( IndexChecker, IndexChecker )
{
  int dims[4] = { 3, 2, 1, 3 };
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, -1,0,0,0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,-1,0,0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,0,-1,0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,0,0,-1 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, dims[0],0,0,0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,dims[1],0,0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,0,dims[2],0 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( indexChecker<4>( dims, 0,0,0,dims[3]), "" );

  for( int a0=0 ; a0<dims[0] ; ++a0 )
  {
    for( int a1=0 ; a1<dims[1] ; ++a1 )
    {
      for( int a2=0 ; a2<dims[2] ; ++a2 )
      {
        for( int a3=0 ; a3<dims[3] ; ++a3 )
        {
          ASSERT_NO_FATAL_FAILURE( indexChecker<4>( dims, a0,a1,a2,a3 ) );
        }
      }
    }
  }
}


TEST( StrideHelper, StrideHelper )
{
  int dims[4] = { 3, 2, 5, 7 };

  int const result0 = stride_helper<4,int>::evaluate(dims);
  int const result1 = stride_helper<3,int>::evaluate(dims+1);
  int const result2 = stride_helper<2,int>::evaluate(dims+2);
  int const result3 = stride_helper<1,int>::evaluate(dims+3);

  ASSERT_TRUE( result0 == dims[1]*dims[2]*dims[3] );
  ASSERT_TRUE( result1 ==         dims[2]*dims[3] );
  ASSERT_TRUE( result2 ==                 dims[3] );
  ASSERT_TRUE( result3 ==                       1 );

}
int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();
  MPI_Finalize();
  return result;
}

#if defined(__APPLE__) &&  defined(__clang__)
#pragma clang diagnostic pop
#endif
