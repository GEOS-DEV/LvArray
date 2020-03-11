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

/**
 * @file tensorOps.hpp
 */

#include "ArraySlice.hpp"

#pragma once

namespace LvArray
{
namespace tensorOps
{

template< typename T, int USD0, int USD1, int USD2, typename INDEX_TYPE >
LVARRAY_HOST_DEVICE constexpr inline
void outerProduct( ArraySlice< T, 2, USD0, INDEX_TYPE > const & a,
                   ArraySlice< T const, 1, USD1, INDEX_TYPE > const & b,
                   ArraySlice< T const, 1, USD2, INDEX_TYPE > const & c )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( b.size(), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    for( INDEX_TYPE j = 0; j < 3; ++j )
    {
      a( i, j ) = b( i ) * c( j );
    }
  }
}

template< typename T, int USD0, int USD1, int USD2, typename INDEX_TYPE >
LVARRAY_HOST_DEVICE constexpr inline
void outerProductPE( ArraySlice< T, 2, USD0, INDEX_TYPE > const & a,
                     ArraySlice< T const, 1, USD1, INDEX_TYPE > const & b,
                     ArraySlice< T const, 1, USD2, INDEX_TYPE > const & c )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( b.size(), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    for( INDEX_TYPE j = 0; j < 3; ++j )
    {
      a( i, j ) = a( i, j ) + b( i ) * c( j );
    }
  }
}

template< typename T, int USD0, int USD1, int USD2, typename INDEX_TYPE >
LVARRAY_HOST_DEVICE constexpr inline
void matTVec( ArraySlice< T, 1, USD0, INDEX_TYPE > const & a,
              ArraySlice< T const, 2, USD1, INDEX_TYPE > const & b,
              ArraySlice< T const, 1, USD2, INDEX_TYPE > const & c )
{
  LVARRAY_ASSERT_EQ( a.size(), 3 );
  LVARRAY_ASSERT_EQ( b.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( b.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  a( 0 ) = b( 0, 0 ) * c( 0 ) + b( 1, 0 ) * c( 1 ) + b( 2, 0 ) * c( 2 );
  a( 1 ) = b( 0, 1 ) * c( 0 ) + b( 1, 1 ) * c( 1 ) + b( 2, 1 ) * c( 2 );
  a( 2 ) = b( 0, 2 ) * c( 0 ) + b( 1, 2 ) * c( 1 ) + b( 2, 2 ) * c( 2 );
}

template< typename T, int USD, typename INDEX_TYPE >
LVARRAY_HOST_DEVICE constexpr inline
T invert( ArraySlice< T, 2, USD, INDEX_TYPE > const & a )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );

  T const temp00 = a( 1, 1 ) * a( 2, 2 ) - a( 1, 2 ) * a( 2, 1 );
  T const temp01 = a( 0, 2 ) * a( 2, 1 ) - a( 0, 1 ) * a( 2, 2 );
  T const temp02 = a( 0, 1 ) * a( 1, 2 ) - a( 0, 2 ) * a( 1, 1 );
  T const temp10 = a( 1, 2 ) * a( 2, 0 ) - a( 1, 0 ) * a( 2, 2 );
  T const temp11 = a( 0, 0 ) * a( 2, 2 ) - a( 0, 2 ) * a( 2, 0 );
  T const temp12 = a( 0, 2 ) * a( 1, 0 ) - a( 0, 0 ) * a( 1, 2 );
  T const temp20 = a( 1, 0 ) * a( 2, 1 ) - a( 1, 1 ) * a( 2, 0 );
  T const temp21 = a( 0, 1 ) * a( 2, 0 ) - a( 0, 0 ) * a( 2, 1 );
  T const temp22 = a( 0, 0 ) * a( 1, 1 ) - a( 0, 1 ) * a( 1, 0 );

  T const det = a( 0, 0 ) * temp00 + a( 1, 0 ) * temp01 + a( 2, 0 ) * temp02;
  T const invDet = 1.0 / det;

  a( 0, 0 ) = temp00 * invDet;
  a( 0, 1 ) = temp01 * invDet;
  a( 0, 2 ) = temp02 * invDet;
  a( 1, 0 ) = temp10 * invDet;
  a( 1, 1 ) = temp11 * invDet;
  a( 1, 2 ) = temp12 * invDet;
  a( 2, 0 ) = temp20 * invDet;
  a( 2, 1 ) = temp21 * invDet;
  a( 2, 2 ) = temp22 * invDet;

  return det;
}

} // namespace tensorOps
} // namespace LvArray
