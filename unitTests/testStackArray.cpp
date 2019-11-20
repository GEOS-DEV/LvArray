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

#include "StackBuffer.hpp"
#include "gtest/gtest.h"
#include "Array.hpp"


using namespace LvArray;

template< typename T, int NDIM, int MAX_SIZE >
using stackArray = StackArray< T, NDIM, camp::make_idx_seq_t< NDIM >, int, MAX_SIZE >;


TEST( StackArray, allocate1d )
{
  stackArray< int, 1, 100 > array1d;
  array1d.resize( 10 );

  for( int i=0 ; i<10 ; ++i )
  {
    array1d[i] = i;
    ASSERT_TRUE( array1d.data()[i] == i );
  }

  ASSERT_DEATH_IF_SUPPORTED( array1d.resize( 101 ), "" );
}

TEST( StackArray, allocate2d )
{
  stackArray< int, 2, 100 > array2d( 10, 10 );

  for( int i=0 ; i<10 ; ++i )
  {
    for( int j=0 ; j<10 ; ++j )
    {
      array2d[i][j] = 10*i+j;

      ASSERT_TRUE( array2d.data()[i*10+j] == 10*i+j );
    }
  }

  ASSERT_DEATH_IF_SUPPORTED( array2d.resize( 11, 10 ), "" );
}

#ifdef USE_ARRAY_BOUNDS_CHECK
TEST( StackArray, BoundsCheck2d )
{
  stackArray< int, 2, 100 > array2d( 10, 10 );

  for( int i=0 ; i<10 ; ++i )
  {
    for( int j=0 ; j<10 ; ++j )
    {
      array2d[i][j] = 10*i+j;
    }
  }

  ASSERT_DEATH_IF_SUPPORTED( array2d[10][11], "" );
}
#endif
