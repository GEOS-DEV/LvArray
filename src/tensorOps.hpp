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

/**
 * @tparam A The type of the matrix @p a.
 * @tparam B The type of the matrix @p b.
 * @tparam C The type of the matrix @p c.
 * @brief Take the outer product of vectors @p b and @p c and write the results to the matrix @p a.
 * @param a A 3x3 matrix to write the results to.
 * @param b A vector of length 3 that is the left operand of the outer product.
 * @param c A vector of length 3 that is the right operand of the outer product.
 */
template< typename A, typename B, typename C >
LVARRAY_HOST_DEVICE constexpr inline
void outerProduct( A const & a, B const & b, C const & c )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( b.size(), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      a( i, j ) = b( i ) * c( j );
    }
  }
}

/**
 * @tparam A The type of the matrix @p a.
 * @tparam B The type of the matrix @p b.
 * @tparam C The type of the matrix @p c.
 * @brief Take the outer product of vectors @p b and @p c and add the results to the matrix @p a.
 * @param a A 3x3 matrix to add the results to.
 * @param b A vector of length 3 that is the left operand of the outer product.
 * @param c A vector of length 3 that is the right operand of the outer product.
 */
template< typename A, typename B, typename C >
LVARRAY_HOST_DEVICE constexpr inline
void outerProductPE( A const & a, B const & b, C const & c )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( b.size(), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      a( i, j ) = a( i, j ) + b( i ) * c( j );
    }
  }
}

/**
 * @tparam A The type of the vector @p a.
 * @tparam B The type of the matrix @p b.
 * @tparam C The type of the vector @p c.
 * @brief Multiply the vector @p c by the transpose of matrix @p b and write the results to the vector @p a.
 * @param a A vector of length 3 to write the results to.
 * @param b A 3x3 matrix.
 * @param c A vector of length 3 that is the multiplied by @p b.
 */
template< typename A, typename B, typename C >
LVARRAY_HOST_DEVICE constexpr inline
void matTVec( A const & a, B const & b, C const & c )
{
  LVARRAY_ASSERT_EQ( a.size(), 3 );
  LVARRAY_ASSERT_EQ( b.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( b.size( 1 ), 3 );
  LVARRAY_ASSERT_EQ( c.size(), 3 );

  a( 0 ) = b( 0, 0 ) * c( 0 ) + b( 1, 0 ) * c( 1 ) + b( 2, 0 ) * c( 2 );
  a( 1 ) = b( 0, 1 ) * c( 0 ) + b( 1, 1 ) * c( 1 ) + b( 2, 1 ) * c( 2 );
  a( 2 ) = b( 0, 2 ) * c( 0 ) + b( 1, 2 ) * c( 1 ) + b( 2, 2 ) * c( 2 );
}

/**
 * @tparam A The type of the matrix @p a.
 * @brief Invert the matrix @p a and @return the determinant.
 * @param a The 3x3 matrix to invert.
 */
template< typename A >
LVARRAY_HOST_DEVICE constexpr inline
auto invert( A const & a )
{
  LVARRAY_ASSERT_EQ( a.size( 0 ), 3 );
  LVARRAY_ASSERT_EQ( a.size( 1 ), 3 );

  auto const temp00 = a( 1, 1 ) * a( 2, 2 ) - a( 1, 2 ) * a( 2, 1 );
  auto const temp01 = a( 0, 2 ) * a( 2, 1 ) - a( 0, 1 ) * a( 2, 2 );
  auto const temp02 = a( 0, 1 ) * a( 1, 2 ) - a( 0, 2 ) * a( 1, 1 );
  auto const temp10 = a( 1, 2 ) * a( 2, 0 ) - a( 1, 0 ) * a( 2, 2 );
  auto const temp11 = a( 0, 0 ) * a( 2, 2 ) - a( 0, 2 ) * a( 2, 0 );
  auto const temp12 = a( 0, 2 ) * a( 1, 0 ) - a( 0, 0 ) * a( 1, 2 );
  auto const temp20 = a( 1, 0 ) * a( 2, 1 ) - a( 1, 1 ) * a( 2, 0 );
  auto const temp21 = a( 0, 1 ) * a( 2, 0 ) - a( 0, 0 ) * a( 2, 1 );
  auto const temp22 = a( 0, 0 ) * a( 1, 1 ) - a( 0, 1 ) * a( 1, 0 );

  auto const det = a( 0, 0 ) * temp00 + a( 1, 0 ) * temp01 + a( 2, 0 ) * temp02;
  auto const invDet = 1.0 / det;

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
