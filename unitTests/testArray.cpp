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
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< typename T >
using array = Array< T, 1 >;

template< typename T >
using array2D = Array< T, 2 >;

template< typename T >
using arrayView = ArrayView< T, 1 >;

template< typename T >
using arrayView2D = ArrayView< T, 2 >;

/**
 * @brief Check that the Array is equivalent to the std::vector. Checks equality using the
 * operator[], operator(), the iterator interface and the raw pointer.
 * @param [in] v the Array to check.
 * @param [in] v_ref the std::vector to check against.
 */
template< class T >
void compare_to_reference( const array< T > & v, const std::vector< T > & v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], v_ref[ i ] );
    ASSERT_EQ( v( i ), v_ref[ i ] );
  }

  const T * v_ptr = v.data();
  const T * ref_ptr = v_ref.data();
  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v_ptr[ i ], ref_ptr[ i ] );
  }
}

/**
 * @brief Check that the Array is equivalent to the std::vector. Checks equality using the
 * operator[], operator(), the iterator interface and the raw pointer.
 * @param [in] v the Array to check.
 * @param [in] v_ref the std::vector to check against.
 */
template< class T >
void compare_to_reference( const array< array< T > > & v,
                           const std::vector< std::vector< T > > & v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    compare_to_reference( v[ i ], v_ref[ i ] );
  }
}

template< class T >
void compare_to_view( array< T > const & v, arrayView< T > const & v_view )
{
  ASSERT_EQ( v.size(), v_view.size() );
  ASSERT_EQ( v.empty(), v_view.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], v_view[ i ] );
    ASSERT_EQ( v( i ), v_view( i ) );
  }

  const T * v_ptr = v.data();
  const T * ref_ptr = v_view.data();
  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v_ptr[ i ], ref_ptr[ i ] );
  }
}

template< class T >
void compare_to_view( array2D< T > const & v, arrayView2D< T > const & v_view )
{
  ASSERT_EQ( v.size(), v_view.size() );
  ASSERT_EQ( v.size( 0 ), v_view.size( 0 ) );
  ASSERT_EQ( v.size( 1 ), v_view.size( 1 ) );
  ASSERT_EQ( v.empty(), v_view.empty() );
  if( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  INDEX_TYPE pos = 0;
  const T * v_ptr = v.data();
  const T * ref_ptr = v_view.data();
  for( INDEX_TYPE i = 0 ; i < v.size( 0 ) ; ++i )
  {
    const T * v_ptr_cur = v.data();
    const T * ref_ptr_cur = v_view.data();
    for( INDEX_TYPE j = 0 ; j < v.size( 1 ) ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], v_view[ i ][ j ] );
      ASSERT_EQ( v( i, j ), v_view( i, j ) );
      ASSERT_EQ( v_ptr[ pos ], ref_ptr[ pos ] );
      ASSERT_EQ( v_ptr_cur[ j ], ref_ptr_cur[ j ] );
      ++pos;
    }
  }
}


template< class T >
void create_2D_test( array2D< T > & v, INDEX_TYPE N, INDEX_TYPE M )
{
  EXPECT_TRUE( v.empty() );

  v.resize( N, M );
  EXPECT_EQ( v.size(), N * M );
  EXPECT_EQ( v.size( 0 ), N );
  EXPECT_EQ( v.size( 1 ), M );

  INDEX_TYPE pos = 0;
  for( INDEX_TYPE i = 0 ; i < N ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < M ; ++j )
    {
      v[i][j] = T( pos );
      pos++;
    }
  }

  pos = 0;
  T const * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < N ; ++i )
  {
    T const * cur_data_ptr = v.data( i );
    for( INDEX_TYPE j = 0 ; j < M ; ++j )
    {
      const T value = T( pos );
      EXPECT_EQ( v[i][j], value );
      EXPECT_EQ( v( i, j ), value );
      EXPECT_EQ( data_ptr[ pos ], value );
      EXPECT_EQ( cur_data_ptr[ j ], value );
      ++pos;
    }
  }
}

/**
 * @brief Test the push_back method of the Array.
 * @param [in/out] v the Array to check.
 * @param [in] n the number of values to append.
 * @return the std::vector compared against.
 */
template< class T >
std::vector< T > push_back_test( array< T > & v, INDEX_TYPE n )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    const T & val = T( i );
    v.push_back( val );
    v_ref.push_back( val );
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the push_back method of the Array.
 * @param [in/out] v the Array to check.
 * @param [in] n the number of values to append.
 * @return the std::vector compared against.
 */
template< class T >
std::vector< std::vector< T > >
push_back_array_test( array< array< T > > & v, INDEX_TYPE n, INDEX_TYPE m )
{
  EXPECT_TRUE( v.empty() );

  std::vector< std::vector< T > > v_ref;
  array< T > v_append( m );
  std::vector< T > v_ref_append( m );
  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      const T val = T( m * i + j );
      v_append[ j ] = val;
      v_ref_append[ j ] = val;
    }

    v.push_back( v_append );
    v_ref.push_back( v_ref_append );
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the insert method of the ChaiVector by inserting multiple values at once.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of insertions to do.
 * @param [in] m the number of values to insert per iteration.
 * @return the std::vector compared against.
 */
template< class T >
std::vector< T > insert_test( array< T > & v, INDEX_TYPE n, INDEX_TYPE m )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  std::vector< T > v_insert( m );
  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      v_insert[ j ] = T( m * i + j );
    }

    if( i % 3 == 0 )    /* Insert at the beginning. */
    {
      v.insert( 0, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin(), v_insert.begin(), v_insert.end() );
    }
    else if( i % 3 == 1 )   /* Insert at the end. */
    {
      v.insert( v.size(), v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.end(), v_insert.begin(), v_insert.end() );
    }
    else  /* Insert in the middle. */
    {
      v.insert( v.size() / 2, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin() + v_ref.size() / 2, v_insert.begin(), v_insert.end() );
    }
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the insert method of the ChaiVector by inserting multiple values at once.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of insertions to do.
 * @param [in] m the number of values to insert per iteration.
 * @return the std::vector compared against.
 */
template< class T >
std::vector< std::vector< T > >
insert_array_test( array< array< T > > & v, INDEX_TYPE n, INDEX_TYPE m, INDEX_TYPE p )
{
  EXPECT_TRUE( v.empty() );

  std::vector< std::vector< T > > v_ref;
  array< array< T > > v_insert;
  std::vector< std::vector< T > > v_ref_insert;
  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    v_insert.clear();
    v_ref_insert.clear();

    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      array< T > temp( p );
      std::vector< T > temp_ref( p );
      for( INDEX_TYPE k = 0 ; k < p ; ++k )
      {
        const T val = T( m * p * i + p * j + k );
        temp[ k ] = val;
        temp_ref[ k ] = val;
      }
      v_insert.push_back( temp );
      v_ref_insert.push_back( temp_ref );
    }

    if( i % 3 == 0 )    /* Insert at the beginning. */
    {
      v.insert( 0, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin(), v_ref_insert.begin(), v_ref_insert.end() );
    }
    else if( i % 3 == 1 )   /* Insert at the end. */
    {
      v.insert( v.size(), v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.end(), v_ref_insert.begin(), v_ref_insert.end() );
    }
    else  /* Insert in the middle. */
    {
      v.insert( v.size() / 2, v_insert.data(), v_insert.size() );
      v_ref.insert( v_ref.begin() + v_ref.size() / 2, v_ref_insert.begin(), v_ref_insert.end() );
    }
  }
  compare_to_reference( v, v_ref );
  return v_ref;
}

/**
 * @brief Test the erase method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to compare against.
 */
template< class T, class U >
void erase_test( array< T > & v, std::vector< U > & v_ref )
{
  const INDEX_TYPE n_elems = v.size();
  for( INDEX_TYPE i = 0 ; i < n_elems ; ++i )
  {
    if( i % 3 == 0 )    /* erase the beginning. */
    {
      v.erase( 0 );
      v_ref.erase( v_ref.begin() );
    }
    else if( i % 3 == 1 )   /* erase at the end. */
    {
      v.erase( v.size() - 1 );
      v_ref.erase( v_ref.end() - 1 );
    }
    else  /* erase the middle. */
    {
      v.erase( v.size() / 2 );
      v_ref.erase( v_ref.begin() + v_ref.size() / 2 );
    }

    if( i % 10 == 0 )
    {
      compare_to_reference( v, v_ref );
    }
  }

  ASSERT_TRUE( v.empty() );
  compare_to_reference( v, v_ref );
}

/**
 * @brief Test the pop_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to compare against.
 */
template< class T, class U >
void pop_back_test( array< T > & v, std::vector< U > & v_ref )
{
  const INDEX_TYPE n_elems = v.size();
  for( INDEX_TYPE i = 0 ; i < n_elems ; ++i )
  {
    v.pop_back();
    v_ref.pop_back();

    if( i % 10 == 0 )
    {
      compare_to_reference( v, v_ref );
    }
  }

  ASSERT_TRUE( v.empty() );
  compare_to_reference( v, v_ref );
}

/**
 * @brief Test the resize method of the Array.
 * @param [in/out] v the Array to check.
 * @param [in] n the end size of the vector.
 */
template< class T >
void resize_test( array< T > & v, INDEX_TYPE const n )
{
  ASSERT_TRUE( v.empty() );

  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    ASSERT_EQ( data_ptr[ i ], T() );
    const T val = T( i );
    data_ptr[ i ] = val;
  }

  /* No reallocation should have occurred. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resize( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for( INDEX_TYPE i = 0 ; i < n / 4 ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }

  v.resize( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    const T val = T( 2 * i );
    v[ i ] = val;
  }

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ], T( 2 * i ) );
  }
}

template< class T >
void resizeNoInitOrDestroy_test( array< T > & v, INDEX_TYPE n )
{
  ASSERT_TRUE( v.empty() );

  v.resizeWithoutInitializationOrDestruction( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    const T val = T( i );
    data_ptr[ i ] = val;
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resizeWithoutInitializationOrDestruction( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for( INDEX_TYPE i = 0 ; i < n / 4 ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }

  v.resizeWithoutInitializationOrDestruction( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    const T val = T( 2 * i );
    v[ i ] = val;
  }

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ], T( 2 * i ) );
  }
}

/**
 * @brief Test the resize method of the Array.
 * @param [in/out] v the Array to check.
 * @param [in] n the end size of the vector.
 */
template< class T >
void resize_array_test( array< array< T > > & v, INDEX_TYPE n, INDEX_TYPE m )
{
  ASSERT_TRUE( v.empty() );

  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  array< T > * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      data_ptr[ i ].push_back( T( m * i + j ) );
    }
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resize( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for( INDEX_TYPE i = 0 ; i < n / 4 ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( m * i + j ) );
    }
  }

  v.resize( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for( INDEX_TYPE i = 0 ; i < n / 4 ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      v[ i ][ j ] = T( 2 * ( m * i + j ) );
    }
  }

  for( INDEX_TYPE i = n / 4 ; i < n ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      v[ i ].push_back( T( 2 * ( m * i + j ) ) );
    }
  }

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ].size(), m );
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( 2 * ( m * i + j ) ) );
    }
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 */
template< class T >
void reserve_test( array< T > & v, INDEX_TYPE n )
{
  ASSERT_TRUE( v.empty() );

  v.reserve( n / 2 );

  ASSERT_EQ( v.size(), 0 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    v.push_back( T( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.reserve( n );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n );

  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }

  data_ptr = v.data();
  for( INDEX_TYPE i = n / 2 ; i < n ; ++i )
  {
    v.push_back( T ( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    ASSERT_EQ( v[ i ], T( i ) );
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 */
template< class T >
void reserve_array_test( array< array< T > > & v, INDEX_TYPE n, INDEX_TYPE m )
{
  ASSERT_TRUE( v.empty() );

  v.reserve( n / 2 );

  ASSERT_EQ( v.size(), 0 );
  ASSERT_EQ( v.capacity(), n / 2 );

  array< T > * data_ptr = v.data();
  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    array< T > temp( m );
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      temp[ j ] = T( m * i + j );
    }

    v.push_back( temp );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.reserve( n );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n );

  for( INDEX_TYPE i = 0 ; i < n / 2 ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( m * i + j ) );
    }
  }

  data_ptr = v.data();
  for( INDEX_TYPE i = n / 2 ; i < n ; ++i )
  {
    array< T > temp( m );
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      temp[ j ] = T( m * i + j );
    }

    v.push_back( temp );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < m ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( m * i + j ) );
    }
  }
}

/**
 * @brief Test the deep_copy method of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 */
template< class T >
void deep_copy_test( const array< T > & v )
{
  array< T > v_cpy( v );

  ASSERT_EQ( v.size(), v_cpy.size() );

  ASSERT_NE( v.data(), v_cpy.data() );

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], v_cpy[ i ] );
    ASSERT_EQ( v[ i ], T( i ) );
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    v_cpy[ i ] = T( 2 * i );
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v_cpy[ i ], T( 2 * i ) );
    ASSERT_EQ( v[ i ], T( i ) );
  }
}

/**
 * @brief Test the deep_copy method of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 */
template< class T >
void deep_copy_array_test( const array< array< T > > & v )
{
  array< array< T > > v_cpy( v );

  ASSERT_EQ( v.size(), v_cpy.size() );

  ASSERT_NE( v.data(), v_cpy.data() );

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

    for( INDEX_TYPE j = 0 ; j < v[ i ].size() ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], v_cpy[ i ][ j ] );
      ASSERT_EQ( v[ i ][ j ], T( v[ i ].size() * i + j ) );
    }
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < v[ i ].size() ; ++j )
    {
      v_cpy[ i ][ j ] = T( 2 * ( v[ i ].size() * i + j ) );
    }
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

    for( INDEX_TYPE j = 0 ; j < v[ i ].size() ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( v[ i ].size() * i + j ) );
      ASSERT_EQ( v_cpy[ i ][ j ], T( 2 * ( v[ i ].size() * i + j ) ) );
    }
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 */
template< class T >
void shallow_copy_test( const array< T > & v )
{
  {
    arrayView< T > v_cpy( v.toView() );
    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.data(), v_cpy.data() );

    for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
    {
      ASSERT_EQ( v[ i ], v_cpy[ i ] );
      ASSERT_EQ( v[ i ], T( i ) );
      v_cpy[ i ] = T( 2 * i );
    }
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    ASSERT_EQ( v[ i ], T( 2 * i ) );
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 */
template< class T >
void shallow_copy_array_test( const array< array< T > > & v )
{
  {
    arrayView< array< T > > v_cpy( static_cast< arrayView< array< T > > const & >(v) );
    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.data(), v_cpy.data() );

    for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
    {
      ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

      for( INDEX_TYPE j = 0 ; j < v[ i ].size() ; ++j )
      {
        ASSERT_EQ( v[ i ][ j ], v_cpy[ i ][ j ] );
        ASSERT_EQ( v[ i ][ j ], T( v[ i ].size() * i + j ) );
        v_cpy[ i ][ j ] = T( 2 * ( v[ i ].size() * i + j ) );
      }
    }
  }

  for( INDEX_TYPE i = 0 ; i < v.size() ; ++i )
  {
    for( INDEX_TYPE j = 0 ; j < v[ i ].size() ; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], T( 2 * ( v[ i ].size() * i + j ) ) );
    }
  }
}

TEST( Array, push_back )
{
  constexpr INDEX_TYPE N = 1000;     /* Number of values to push_back */

  {
    array< int > v;
    push_back_test( v, N );
  }

  {
    array< Tensor > v;
    push_back_test( v, N );
  }

  {
    array< TestString > v;
    push_back_test( v, N );
  }
}

TEST( Array, push_back_array )
{
  constexpr INDEX_TYPE N = 100;      /* Number of arrays to push_back */
  constexpr INDEX_TYPE M = 10;       /* Size of each array */

  {
    array< array< int > > v;
    push_back_array_test( v, N, M );
  }

  {
    array< array< Tensor > > v;
    push_back_array_test( v, N, M );
  }

  {
    array< array< TestString > > v;
    push_back_array_test( v, N, M );
  }
}

TEST( Array, insert )
{
  constexpr INDEX_TYPE N = 100;      /* Number of times to call insert */
  constexpr INDEX_TYPE M = 10;       /* Number of values inserted at each call */

  {
    array< int > v;
    insert_test( v, N, M );
  }

  {
    array< Tensor > v;
    insert_test( v, N, M );
  }

  {
    array< TestString > v;
    insert_test( v, N, M );
  }
}

TEST( Array, insert_array )
{
  constexpr INDEX_TYPE N = 10;       /* Number of times to call insert */
  constexpr INDEX_TYPE M = 10;       /* Number of arrays inserted at each call */
  constexpr INDEX_TYPE P = 10;       /* Size of each array inserted */

  {
    array< array< int > > v;
    insert_array_test( v, N, M, P );
  }

  {
    array< array< Tensor > > v;
    insert_array_test( v, N, M, P );
  }

  {
    array< array< TestString > > v;
    insert_array_test( v, N, M, P );
  }
}

TEST( Array, erase )
{
  constexpr INDEX_TYPE N = 200;    /* Size of the array */

  {
    array< int > v;
    std::vector< int > v_ref = push_back_test( v, N );
    erase_test( v, v_ref );
  }

  {
    array< Tensor > v;
    std::vector< Tensor > v_ref =  push_back_test( v, N );
    erase_test( v, v_ref );
  }

  {
    array< TestString > v;
    std::vector< TestString > v_ref =  push_back_test( v, N );
    erase_test( v, v_ref );
  }

}

TEST( Array, erase_array )
{
  constexpr INDEX_TYPE N = 100;    /* Number of arrays to push_back */
  constexpr INDEX_TYPE M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    std::vector< std::vector< int > > v_ref = push_back_array_test( v, N, M );
    erase_test( v, v_ref );
  }

  {
    array< array< Tensor > > v;
    std::vector< std::vector< Tensor > > v_ref =  push_back_array_test( v, N, M );
    erase_test( v, v_ref );
  }

  {
    array< array< TestString > > v;
    std::vector< std::vector< TestString > > v_ref =  push_back_array_test( v, N, M );
    erase_test( v, v_ref );
  }
}

TEST( Array, pop_back )
{
  constexpr INDEX_TYPE N = 300;    /* Size of the array */

  {
    array< int > v;
    std::vector< int > v_ref = push_back_test( v, N );
    pop_back_test( v, v_ref );
  }

  {
    array< Tensor > v;
    std::vector< Tensor > v_ref =  push_back_test( v, N );
    pop_back_test( v, v_ref );
  }

  {
    array< TestString > v;
    std::vector< TestString > v_ref =  push_back_test( v, N );
    pop_back_test( v, v_ref );
  }
}

TEST( Array, pop_back_array )
{
  constexpr INDEX_TYPE N = 50;     /* Number of arrays to push_back */
  constexpr INDEX_TYPE M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    std::vector< std::vector< int > > v_ref = push_back_array_test( v, N, M );
    pop_back_test( v, v_ref );
  }

  {
    array< array< Tensor > > v;
    std::vector< std::vector< Tensor > > v_ref =  push_back_array_test( v, N, M );
    pop_back_test( v, v_ref );
  }

  {
    array< array< TestString > > v;
    std::vector< std::vector< TestString > > v_ref =  push_back_array_test( v, N, M );
    pop_back_test( v, v_ref );
  }
}

TEST( Array, resize )
{
  constexpr INDEX_TYPE N = 1000;   /* Size of each array */

  {
    array< int > v;
    resize_test( v, N );
  }

  {
    array< Tensor > v;
    resize_test( v, N );
  }

  {
    array< TestString > v;
    resize_test( v, N );
  }
}

TEST( Array, resizeNoInitOrDestroy )
{
  constexpr INDEX_TYPE N = 1000;   /* Size of each array */

  {
    array< int > v;
    resizeNoInitOrDestroy_test( v, N );
  }

  {
    array< Tensor > v;
    resizeNoInitOrDestroy_test( v, N );
  }
}

TEST( Array, resize_array )
{
  constexpr INDEX_TYPE N = 100;    /* Size of each array */
  constexpr INDEX_TYPE M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    resize_array_test( v, N, M );
  }

  {
    array< array< Tensor > > v;
    resize_array_test( v, N, M );
  }

  {
    array< array< TestString > > v;
    resize_array_test( v, N, M );
  }
}

TEST( Array, reserve )
{
  constexpr INDEX_TYPE N = 1000;   /* Size of the array */

  {
    array< int > v;
    reserve_test( v, N );
  }

  {
    array< Tensor > v;
    reserve_test( v, N );
  }

  {
    array< TestString > v;
    reserve_test( v, N );
  }
}

TEST( Array, reserve_array )
{
  constexpr INDEX_TYPE N = 100;    /* Number of arrays */
  constexpr INDEX_TYPE M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    reserve_array_test( v, N, M );
  }

  {
    array< array< Tensor > > v;
    reserve_array_test( v, N, M );
  }

  {
    array< array< TestString > > v;
    reserve_array_test( v, N, M );
  }
}

TEST( Array, deep_copy )
{
  constexpr INDEX_TYPE N = 1000;   /* Size of the array */

  {
    array< int > v;
    push_back_test( v, N );
    deep_copy_test( v );
  }

  {
    array< Tensor > v;
    push_back_test( v, N );
    deep_copy_test( v );
  }

  {
    array< TestString > v;
    push_back_test( v, N );
    deep_copy_test( v );
  }
}

TEST( Array, deep_copy_array )
{
  constexpr INDEX_TYPE N = 100;    /* Number of arrays */
  constexpr INDEX_TYPE M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    push_back_array_test( v, N, M );
    deep_copy_array_test( v );
  }

  {
    array< array< Tensor > > v;
    push_back_array_test( v, N, M );
    deep_copy_array_test( v );
  }

  {
    array< array< TestString > > v;
    push_back_array_test( v, N, M );
    deep_copy_array_test( v );
  }
}

TEST( Array, shallow_copy )
{
  constexpr INDEX_TYPE N = 1000;  /* Size of the array */

  {
    array< int > v;
    push_back_test( v, N );
    shallow_copy_test( v );
  }

  {
    array< Tensor > v;
    push_back_test( v, N );
    shallow_copy_test( v );
  }

  {
    array< TestString > v;
    push_back_test( v, N );
    shallow_copy_test( v );
  }
}

TEST( Array, shallow_copy_array )
{
  constexpr INDEX_TYPE N = 100;   /* Number of arrays */
  constexpr INDEX_TYPE M = 10;    /* Size of each array */

  {
    array< array< int > > v;
    push_back_array_test( v, N, M );
    shallow_copy_array_test( v );
  }

  {
    array< array< Tensor > > v;
    push_back_array_test( v, N, M );
    shallow_copy_array_test( v );
  }

  {
    array< array< TestString > > v;
    push_back_array_test( v, N, M );
    shallow_copy_array_test( v );
  }
}

TEST( Array, test_upcast )
{
  constexpr INDEX_TYPE N = 1000;     /* Number of values to push_back */

  {
    array< int > v;
    push_back_test( v, N );
    arrayView< int > & vView = v;
    compare_to_view( v, vView );
  }

  {
    array< Tensor > v;
    push_back_test( v, N );
    arrayView< Tensor > & vView = v;
    compare_to_view( v, vView );
  }

  {
    array< TestString > v;
    push_back_test( v, N );
    arrayView< TestString > & vView = v;
    compare_to_view( v, vView );
  }
}

TEST( Array, test_array2D )
{
  constexpr INDEX_TYPE N = 53;
  constexpr INDEX_TYPE M = 47;

  {
    array2D< int > v;
    create_2D_test( v, N, M );
    arrayView2D< int > & vView = v;
    compare_to_view( v, vView );
  }

  {
    array2D< Tensor > v;
    create_2D_test( v, N, M );
    arrayView2D< Tensor > & vView = v;
    compare_to_view( v, vView );
  }

  {
    array2D< TestString > v;
    create_2D_test( v, N, M );
    arrayView2D< TestString > & vView = v;
    compare_to_view( v, vView );
  }
}

template< typename T >
class ArrayResizeTest : public ::testing::Test
{
public:

  template< int UNIT_STRIDE_DIM >
  void print( ArrayView< T const, 2, UNIT_STRIDE_DIM > const & v )
  {
    for( INDEX_TYPE i = 0 ; i < v.size( 0 ) ; ++i )
    {
      for( INDEX_TYPE j = 0 ; j < v.size( 1 ) ; ++j )
      {
        std::cout << v( i, j ) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  template< typename PERMUTATION >
  void resize( PERMUTATION )
  {
    constexpr int NDIM = getDimension( PERMUTATION {} );
    static_assert( NDIM < 5, "dimension must be less than 4" );

    Array< T, NDIM, PERMUTATION > a;

    INDEX_TYPE const maxDimSize = getMaxDimSize( NDIM );
    INDEX_TYPE initialSizes[ NDIM ];

    // Iterate over randomly generated sizes.
    for( int i = 0 ; i < 10 ; ++i )
    {
      for( int d = 0 ; d < NDIM ; ++d )
      {
        initialSizes[ d ] = rand( 1, maxDimSize / 2 );
      }

      // Iterate over the dimensions
      for( int d = 0 ; d < NDIM ; ++d )
      {
        a.setSingleParameterResizeIndex( d );

        populate( a, initialSizes );

        // Increase the size
        INDEX_TYPE const largerDefaultSize = rand( initialSizes[ d ], maxDimSize );
        a.resizeDefault( largerDefaultSize, T( -1 ) );
        validate( a.toViewConst(), initialSizes, T( -1 ) );

        // Decrease the size
        INDEX_TYPE const smallerDefaultSize = rand( 0, initialSizes[ d ] );
        a.resize( smallerDefaultSize );
        validate( a.toViewConst(), a.dims() );
      }
    }
  }


private:

  INDEX_TYPE getTestingLinearIndex( INDEX_TYPE const i )
  {
    return i;
  }

  INDEX_TYPE getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j )
  {
    INDEX_TYPE const maxDimSize = getMaxDimSize( 2 );
    return maxDimSize * i + j;
  }

  INDEX_TYPE getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const k )
  {
    INDEX_TYPE const maxDimSize = getMaxDimSize( 3 );
    return maxDimSize * maxDimSize * i + maxDimSize * j + k;
  }

  INDEX_TYPE getTestingLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const k, INDEX_TYPE const l )
  {
    INDEX_TYPE const maxDimSize = getMaxDimSize( 4 );
    return maxDimSize * maxDimSize * maxDimSize * i + maxDimSize * maxDimSize * j + maxDimSize * k + l;
  }


  template< int NDIM, typename PERMUTATION >
  void populate( Array< T, NDIM, PERMUTATION > & a,
                 INDEX_TYPE const * const initialSizes )
  {
    a.resize( NDIM, initialSizes );
    for( int d = 0 ; d < NDIM ; ++d )
    {
      ASSERT_EQ( initialSizes[ d ], a.size( d ) );
    }

    forValuesInSliceWithIndices( a.toSlice(),
                                 [this]( T & value, auto const... indices )
        {
          INDEX_TYPE const idx = this->getTestingLinearIndex( indices ... );
          value = T( idx );
        }
                                 );

    validate( a.toViewConst(), initialSizes );
  }

  template< int NDIM, int UNIT_STRIDE_DIM >
  void validate( ArrayView< T const, NDIM, UNIT_STRIDE_DIM > const & v,
                 INDEX_TYPE const * const initialSizes,
                 T const & defaultValue = T() )
  {
    forValuesInSliceWithIndices( v.toSlice(),
                                 [initialSizes, defaultValue, this]( T const & value, auto const... indices )
        {
          if( !invalidIndices( initialSizes, indices ... ) )
          {
            INDEX_TYPE const idx = this->getTestingLinearIndex( indices ... );
            EXPECT_EQ( value, T( idx ) );
          }
          else
          {
            EXPECT_EQ( value, defaultValue );
          }
        }
                                 );
  }

  INDEX_TYPE getMaxDimSize( int const d )
  { return m_maxDimSizes[ d - 1 ]; }

  INDEX_TYPE rand( INDEX_TYPE const min, INDEX_TYPE const max )
  { return std::uniform_int_distribution< INDEX_TYPE >( min, max )( m_gen ); }

  INDEX_TYPE const m_maxDimSizes[ 4 ] = { 1000, 40, 20, 10 };
  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types< INDEX_TYPE, Tensor, TestString >;
TYPED_TEST_CASE( ArrayResizeTest, TestTypes );

TYPED_TEST( ArrayResizeTest, resize1D )
{
  this->resize( RAJA::PERM_I {} );
}

TYPED_TEST( ArrayResizeTest, resize2D )
{
  this->resize( RAJA::PERM_IJ {} );
  this->resize( RAJA::PERM_JI {} );
}

TYPED_TEST( ArrayResizeTest, resize3D )
{
  this->resize( RAJA::PERM_IJK {} );
  this->resize( RAJA::PERM_JIK {} );
  this->resize( RAJA::PERM_IKJ {} );
  this->resize( RAJA::PERM_KIJ {} );
  this->resize( RAJA::PERM_KJI {} );
  this->resize( RAJA::PERM_JKI {} );
}

TYPED_TEST( ArrayResizeTest, resize4D )
{
  this->resize( RAJA::PERM_IJKL {} );
  this->resize( RAJA::PERM_JIKL {} );
  this->resize( RAJA::PERM_IKJL {} );
  this->resize( RAJA::PERM_KIJL {} );
  this->resize( RAJA::PERM_JKIL {} );
  this->resize( RAJA::PERM_KJIL {} );
  this->resize( RAJA::PERM_IJLK {} );
  this->resize( RAJA::PERM_JILK {} );
  this->resize( RAJA::PERM_ILJK {} );
  this->resize( RAJA::PERM_LIJK {} );
  this->resize( RAJA::PERM_JLIK {} );
  this->resize( RAJA::PERM_LJIK {} );
  this->resize( RAJA::PERM_IKLJ {} );
  this->resize( RAJA::PERM_KILJ {} );
  this->resize( RAJA::PERM_ILKJ {} );
  this->resize( RAJA::PERM_LIKJ {} );
  this->resize( RAJA::PERM_KLIJ {} );
  this->resize( RAJA::PERM_LKIJ {} );
  this->resize( RAJA::PERM_JKLI {} );
  this->resize( RAJA::PERM_KJLI {} );
  this->resize( RAJA::PERM_JLKI {} );
  this->resize( RAJA::PERM_LJKI {} );
  this->resize( RAJA::PERM_KLJI {} );
  this->resize( RAJA::PERM_LKJI {} );
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
