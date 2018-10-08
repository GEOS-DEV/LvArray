/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include <string>

#include "Array.hpp"

using namespace LvArray;

template < typename T >
using array = Array< T, 1 >;

namespace internal
{

/**
 * @brief Check that the Array is equivalent to the std::vector. Checks equality using the
 * operator[], operator(), the iterator interface and the raw pointer.
 * @param [in] v the Array to check.
 * @param [in] v_ref the std::vector to check against. 
 */
template < class T >
void compare_to_reference( const array< T >& v, const std::vector< T >& v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if ( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], v_ref[ i ] );
    ASSERT_EQ( v( i ), v_ref[ i ] );
  }

  ASSERT_EQ( v.front(), v_ref.front() );
  ASSERT_EQ( v.back(), v_ref.back() );

  typename array< T >::const_iterator it = v.begin();
  typename std::vector< T >::const_iterator ref_it = v_ref.begin();
  for ( ; it != v.end(); ++it )
  {
    ASSERT_EQ( *it, *ref_it );
    ++ref_it;
  }

  const T* v_ptr = v.data();
  const T* ref_ptr = v_ref.data();
  for ( int i = 0; i < v.size(); ++i )
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
template < class T >
void compare_to_reference( const array< array< T > >& v, 
                           const std::vector< std::vector< T > >& v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if ( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    compare_to_reference( v[ i ], v_ref[ i ] );
  }
}

/**
 * @brief Test the push_back method of the Array.
 * @param [in/out] v the Array to check.
 * @param [in] n the number of values to append.
 * @param [in] get_value a function to generate the values to append.
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< T > push_back_test( array< T >& v, int n, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  for ( int i = 0; i < n; ++i )
  {
    const T& val = get_value( i );
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
 * @param [in] get_value a function to generate the values to append.
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< std::vector< T > > 
push_back_array_test( array< array < T > >& v, int n, int m, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );

  std::vector< std::vector< T > > v_ref;
  array< T > v_append( m );
  std::vector< T > v_ref_append( m );
  for ( int i = 0; i < n; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      const T val = get_value( m * i + j );
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
 * @param [in] get_value a function to generate the values to insert. 
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< T > insert_test( array< T >& v, int n, int m, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );
  
  std::vector< T > v_ref;
  std::vector< T > v_insert( m );
  for ( int i = 0; i < n; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      v_insert[ j ] = get_value( m * i + j );
    }

    if ( i % 3 == 0 )   /* Insert at the beginning. */
    {
      v.insert( v.begin(), v_insert.begin(), v_insert.end() );
      v_ref.insert( v_ref.begin(), v_insert.begin(), v_insert.end() );
    }
    else if ( i % 3 == 1 )  /* Insert at the end. */
    {
      v.insert( v.end(), v_insert.begin(), v_insert.end() );
      v_ref.insert( v_ref.end(), v_insert.begin(), v_insert.end() );
    }
    else  /* Insert in the middle. */
    {
      v.insert( v.begin() + v.size() / 2, v_insert.begin(), v_insert.end() );
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
 * @param [in] get_value a function to generate the values to insert. 
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< std::vector< T > >
insert_array_test( array< array< T > >& v, int n, int m, int p, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );
  
  std::vector< std::vector< T > > v_ref;
  array< array< T > > v_insert;
  std::vector< std::vector< T > > v_ref_insert;
  for ( int i = 0; i < n; ++i )
  {
    v_insert.clear();
    v_ref_insert.clear();

    for ( int j = 0; j < m; ++j )
    {
      array< T > temp( p );
      std::vector< T > temp_ref( p );
      for ( int k = 0; k < p; ++k )
      {
        const T val = get_value( m * p * i + p * j + k );
        temp[ k ] = val;
        temp_ref[ k ] = val;
      }
      v_insert.push_back( temp );
      v_ref_insert.push_back( temp_ref );
    }

    if ( i % 3 == 0 )   /* Insert at the beginning. */
    {
      v.insert( v.begin(), v_insert.begin(), v_insert.end() );
      v_ref.insert( v_ref.begin(), v_ref_insert.begin(), v_ref_insert.end() );
    }
    else if ( i % 3 == 1 )  /* Insert at the end. */
    {
      v.insert( v.end(), v_insert.begin(), v_insert.end() );
      v_ref.insert( v_ref.end(), v_ref_insert.begin(), v_ref_insert.end() );
    }
    else  /* Insert in the middle. */
    {
      v.insert( v.begin() + v.size() / 2, v_insert.begin(), v_insert.end() );
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
template < class T, class U >
void erase_test( array< T >& v, std::vector< U >& v_ref )
{
  const int n_elems = v.size();
  for ( int i = 0; i < n_elems; ++i )
  {
    if ( i % 3 == 0 )   /* erase the beginning. */
    {
      v.erase( v.begin() );
      v_ref.erase( v_ref.begin() );
    }
    else if ( i % 3 == 1 )  /* erase at the end. */
    {
      v.erase( v.end() - 1 );
      v_ref.erase( v_ref.end() - 1 );
    }
    else  /* erase the middle. */
    {
      v.erase( v.begin() + v.size() / 2 );
      v_ref.erase( v_ref.begin() + v_ref.size() / 2 );
    }

    if ( i % 10 == 0 )
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
template < class T, class U >
void pop_back_test( array< T >& v, std::vector< U >& v_ref )
{
  const int n_elems = v.size();
  for ( int i = 0; i < n_elems; ++i )
  {
    v.pop_back();
    v_ref.pop_back();

    if ( i % 10 == 0 )
    {
      compare_to_reference( v, v_ref );
    }
  }

  ASSERT_TRUE( v.empty() );
  compare_to_reference( v, v_ref );
}

/**
 * @brief Test the resize method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void resize_test( array< T >& v, int n, LAMBDA get_value )
{
  ASSERT_TRUE( v.empty() );
  
  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T* data_ptr = v.data();
  for ( int i = 0; i < n / 2; ++i )
  {
    ASSERT_EQ( data_ptr[ i ], T() );
    const T val = get_value( i );
    data_ptr[ i ] = val;
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resize( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for ( int i = 0; i < n / 4; ++i )
  {
    ASSERT_EQ( v[ i ], get_value( i ) );
  }

  v.resize( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for ( int i = 0; i < n; ++i )
  {
    const T val = get_value( 2 * i );
    v[ i ] = val;
  }

  for ( int i = 0; i < n; ++i )
  {
    ASSERT_EQ( v[ i ], get_value( 2 * i ) );
  }
}

/**
 * @brief Test the resize method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void resize_array_test( array< array< T > >& v, int n, int m, LAMBDA get_value )
{
  ASSERT_TRUE( v.empty() );

  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  array< T >* data_ptr = v.data();
  for ( int i = 0; i < n / 2; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      data_ptr[ i ].push_back( get_value( m * i + j ) );
    }
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.resize( n / 4 );

  ASSERT_EQ( v.size(), n / 4 );
  ASSERT_EQ( v.capacity(), n / 2 );

  for ( int i = 0; i < n / 4; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( m * i + j ) );
    }
  }

  v.resize( n );

  ASSERT_EQ( v.size(), n );
  ASSERT_EQ( v.capacity(), n );

  for ( int i = 0; i < n / 4; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      v[ i ][ j ] = get_value( 2 * ( m * i + j ) );
    }
  }

  for ( int i = n / 4; i < n; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      v[ i ].push_back( get_value( 2 * ( m * i + j ) ) );
    }
  }

  for ( int i = 0; i < n; ++i )
  {
    ASSERT_EQ( v[ i ].size(), m );
    for ( int j = 0; j < m; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( 2 * ( m * i + j ) ) );
    }
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void reserve_test( array< T >& v, int n, LAMBDA get_value )
{
  ASSERT_TRUE( v.empty() );
  
  v.reserve( n / 2 );

  ASSERT_EQ( v.size(), 0 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T* data_ptr = v.data();
  for ( int i = 0; i < n / 2; ++i )
  {
    v.push_back( get_value( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.reserve( n );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n );

  for ( int i = 0; i < n / 2; ++i )
  {
    ASSERT_EQ( v[ i ], get_value( i ) );
  }

  data_ptr = v.data();
  for ( int i = n / 2; i < n; ++i )
  {
    v.push_back( get_value ( i ) );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  for ( int i = 0; i < n; ++i )
  {
    ASSERT_EQ( v[ i ], get_value( i ) );
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void reserve_array_test( array< array< T > >& v, int n, int m, LAMBDA get_value )
{
  ASSERT_TRUE( v.empty() );
  
  v.reserve( n / 2 );

  ASSERT_EQ( v.size(), 0 );
  ASSERT_EQ( v.capacity(), n / 2 );

  array< T >* data_ptr = v.data();
  for ( int i = 0; i < n / 2; ++i )
  {
    array< T > temp( m );
    for ( int j = 0; j < m; ++j )
    {
      temp[ j ] = get_value( m * i + j );
    }

    v.push_back( temp );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  v.reserve( n );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n );

  for ( int i = 0; i < n / 2; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( m * i + j ) );
    }
  }

  data_ptr = v.data();
  for ( int i = n / 2; i < n; ++i )
  {
    array< T > temp( m );
    for ( int j = 0; j < m; ++j )
    {
      temp[ j ] = get_value( m * i + j );
    }

    v.push_back( temp );
  }

  /* No reallocation should have occured. */
  ASSERT_EQ( data_ptr, v.data() );

  for ( int i = 0; i < n; ++i )
  {
    for ( int j = 0; j < m; ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( m * i + j ) );
    }
  }
}

/**
 * @brief Test the deep_copy method of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void deep_copy_test( const array< T >& v, LAMBDA get_value )
{
  array< T > v_cpy;
  v_cpy = v;
  
  ASSERT_EQ( v.size(), v_cpy.size() );

  ASSERT_NE( v.data(), v_cpy.data() );

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], v_cpy[ i ] );
    ASSERT_EQ( v[ i ], get_value( i ) );
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    v_cpy[ i ] = get_value( 2 * i );
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v_cpy[ i ], get_value( 2 * i ) );
    ASSERT_EQ( v[ i ], get_value( i ) );
  }
}

/**
 * @brief Test the deep_copy method of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void deep_copy_array_test( const array< array< T > >& v, LAMBDA get_value )
{
  array< array< T > > v_cpy;
  v_cpy = v;
  
  ASSERT_EQ( v.size(), v_cpy.size() );

  ASSERT_NE( v.data(), v_cpy.data() );

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

    for ( int j = 0; j < v[ i ].size(); ++j )
    {
      ASSERT_EQ( v[ i ][ j ], v_cpy[ i ][ j ] );
      ASSERT_EQ( v[ i ][ j ], get_value( v[ i ].size() * i + j ) );
    }
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    for ( int j = 0; j < v[ i ].size(); ++j )
    {
      v_cpy[ i ][ j ] = get_value( 2 * ( v[ i ].size() * i + j ) );
    }
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

    for ( int j = 0; j < v[ i ].size(); ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( v[ i ].size() * i + j ) );
      ASSERT_EQ( v_cpy[ i ][ j ], get_value( 2 * ( v[ i ].size() * i + j ) ) );
    }
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void shallow_copy_test( const array< T >& v, LAMBDA get_value )
{
  {
    array< T > v_cpy(v);
    
    ASSERT_TRUE( v_cpy.isCopy() );
    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.capacity(), v_cpy.capacity() );

    ASSERT_EQ( v.data(), v_cpy.data() );

    for ( int i = 0; i < v.size(); ++i )
    {
      ASSERT_EQ( v[ i ], v_cpy[ i ] );
      ASSERT_EQ( v[ i ], get_value( i ) );
      v_cpy[ i ] = get_value( 2 * i );
    }
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], get_value( 2 * i ) );
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void shallow_copy_array_test( const array< array< T > >& v, LAMBDA get_value )
{
  {
    array< array< T > > v_cpy(v);
    
    ASSERT_TRUE( v_cpy.isCopy() );
    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.capacity(), v_cpy.capacity() );

    ASSERT_EQ( v.data(), v_cpy.data() );

    for ( int i = 0; i < v.size(); ++i )
    {
      ASSERT_EQ( v[ i ].size(), v_cpy[ i ].size() );

      for ( int j = 0; j < v[ i ].size(); ++j )
      {
        ASSERT_EQ( v[ i ][ j ], v_cpy[ i ][ j ] );
        ASSERT_EQ( v[ i ][ j ], get_value( v[ i ].size() * i + j ) );
        v_cpy[ i ][ j ] = get_value( 2 * ( v[ i ].size() * i + j ) );
      }
    }
  }

  for ( int i = 0; i < v.size(); ++i )
  {
    for ( int j = 0; j < v[ i ].size(); ++j )
    {
      ASSERT_EQ( v[ i ][ j ], get_value( 2 * ( v[ i ].size() * i + j ) ) );
    }
  }
}

} /* namespace internal */


struct Tensor
{
  double x, y, z;

  Tensor():
    x(), y(), z()
  {}

  Tensor( double val ):
    x( val ), y( val ), z( val )
  {}

  bool operator==(const Tensor& other) const
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return x == other.x && y == other.y && z == other.z; 
#pragma GCC diagnostic pop
  }
};

TEST( Array, viewUpcast)
{
  Array<int,2,int> arr( 10, 1 );

  ArrayView<int,2,int> & arrView = arr;

  EXPECT_EQ( arrView[9][0], arr[9][0] );
}

//TEST( Array, test_const )
//{
//  int junk[ 10 ];
//  int dim[ 1 ] = { 10 };
//  int stride[ 1 ] = { 1 };
//
//
//  ArrayView< int, 1, int > array( junk, dim, stride );
//
//  ArrayView< const int, 1, int > arrayC = array;
//
//  EXPECT_TRUE( arrayC[ 9 ] == array[ 9 ] );
//}

TEST( Array, push_back )
{
  constexpr int N = 1000;     /* Number of values to push_back */

  {
    array< int > v;
    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    array< Tensor > v;
    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< std::string > v;
    internal::push_back_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, push_back_array )
{
  constexpr int N = 100;      /* Number of arrays to push_back */
  constexpr int M = 10;       /* Size of each array */

  {
    array< array< int > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> int { return i; } );
  }

  {
    array< array< Tensor > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< array< std::string > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, insert )
{
  constexpr int N = 100;      /* Number of times to call insert */
  constexpr int M = 10;       /* Number of values inserted at each call */

  {
    array< int > v;
    internal::insert_test( v, N, M, []( int i ) -> int { return i; } );
  }

  {
    array< Tensor > v;
    internal::insert_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< std::string > v;
    internal::insert_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, insert_array )
{
  constexpr int N = 10;       /* Number of times to call insert */
  constexpr int M = 10;       /* Number of arrays inserted at each call */
  constexpr int P = 10;       /* Size of each array inserted */

  {
    array< array< int > > v;
    internal::insert_array_test( v, N, M, P, []( int i ) -> int { return i; } );
  }

  {
    array< array< Tensor > > v;
    internal::insert_array_test( v, N, M, P, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< array< std::string > > v;
    internal::insert_array_test( v, N, M, P, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, erase )
{
  constexpr int N = 200;    /* Size of the array */

  {
    array< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::erase_test( v, v_ref );
  }

  {
    array< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::erase_test( v, v_ref );
  }

  {
    array< std::string > v;
    std::vector< std::string > v_ref =  internal::push_back_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::erase_test( v, v_ref );
  }

}

TEST( Array, erase_array )
{
  constexpr int N = 100;    /* Number of arrays to push_back */
  constexpr int M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    std::vector< std::vector< int > > v_ref = internal::push_back_array_test( v, N, M, []( int i ) -> int { return i; } );
    internal::erase_test( v, v_ref );
  }

  {
    array< array< Tensor > > v;
    std::vector< std::vector < Tensor > > v_ref =  internal::push_back_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::erase_test( v, v_ref );
  }

  {
    array< array< std::string > > v;
    std::vector< std::vector < std::string > > v_ref =  internal::push_back_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::erase_test( v, v_ref );
  }
}

TEST( Array, pop_back )
{
  constexpr int N = 300;    /* Size of the array */

  {
    array< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::pop_back_test( v, v_ref );
  }

  {
    array< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::pop_back_test( v, v_ref );
  }

  {
    array< std::string > v;
    std::vector< std::string > v_ref =  internal::push_back_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::pop_back_test( v, v_ref );
  }
}

TEST( Array, pop_back_array )
{
  constexpr int N = 30;     /* Number of arrays to push_back */
  constexpr int M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    std::vector< std::vector< int > > v_ref = internal::push_back_array_test( v, N, M, []( int i ) -> int { return i; } );
    internal::pop_back_test( v, v_ref );
  }

  {
    array< array< Tensor > > v;
    std::vector< std::vector < Tensor > > v_ref =  internal::push_back_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::pop_back_test( v, v_ref );
  }

  {
    array< array< std::string > > v;
    std::vector< std::vector < std::string > > v_ref =  internal::push_back_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::pop_back_test( v, v_ref );
  }
}

TEST( Array, resize )
{
  constexpr int N = 1000;   /* Size of each array */

  {
    array< int > v;
    internal::resize_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    array< Tensor > v;
    internal::resize_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< std::string > v;
    internal::resize_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, resize_array )
{
  constexpr int N = 100;    /* Size of each array */
  constexpr int M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    internal::resize_array_test( v, N, M, []( int i ) -> int { return i; } );
  }

  {
    array< array< Tensor > > v;
    internal::resize_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< array< std::string > > v;
    internal::resize_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, reserve )
{
  constexpr int N = 1000;   /* Size of the array */

  {
    array< int > v;
    internal::reserve_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    array< Tensor > v;
    internal::reserve_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< std::string > v;
    internal::reserve_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, reserve_array )
{
  constexpr int N = 100;    /* Number of arrays */
  constexpr int M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    internal::reserve_array_test( v, N, M, []( int i ) -> int { return i; } );
  }

  {
    array< array< Tensor > > v;
    internal::reserve_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< array< std::string > > v;
    internal::reserve_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, deep_copy )
{
  constexpr int N = 1000;   /* Size of the array */

  {
    array< int > v;
    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::deep_copy_test( v, []( int i ) -> int { return i; } );
  }

  {
    array< Tensor > v;
    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::deep_copy_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
  }

  {
    array< std::string > v;
    internal::push_back_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::deep_copy_test( v, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

TEST( Array, deep_copy_array )
{
  constexpr int N = 100;    /* Number of arrays */
  constexpr int M = 10;     /* Size of each array */

  {
    array< array< int > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> int { return i; } );
    internal::deep_copy_array_test( v, []( int i ) -> int { return i; } );
  }

  {
    array< array< Tensor > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::deep_copy_array_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
  }

{
    array< array< std::string > > v;
    internal::push_back_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
    internal::deep_copy_array_test( v, []( int i ) -> std::string { return std::to_string( i ); } );
  }
}

//TEST( Array, shallow_copy )
//{
//  constexpr int N = 1000;   /* Size of the array */
//
//  {
//    array< int > v;
//    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
//    internal::shallow_copy_test( v, []( int i ) -> int { return i; } );
//  }
//
//  {
//    array< Tensor > v;
//    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
//    internal::shallow_copy_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
//  }
//
//  {
//    array< std::string > v;
//    internal::push_back_test( v, N, []( int i ) -> std::string { return std::to_string( i ); } );
//    internal::shallow_copy_test( v, []( int i ) -> std::string { return std::to_string( i ); } );
//  }
//}

//TEST( Array, shallow_copy_array )
//{
//  constexpr int N = 100;    /* Number of arrays */
//  constexpr int M = 10;     /* Size of each array */
//
//  {
//    array< array< int > > v;
//    internal::push_back_array_test( v, N, M, []( int i ) -> int { return i; } );
//    internal::shallow_copy_array_test( v, []( int i ) -> int { return i; } );
//  }
//
//  {
//    array< array< Tensor > > v;
//    internal::push_back_array_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
//    internal::shallow_copy_array_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
//  }
//
//  {
//    array< array< std::string > > v;
//    internal::push_back_array_test( v, N, M, []( int i ) -> std::string { return std::to_string( i ); } );
//    internal::shallow_copy_array_test( v, []( int i ) -> std::string { return std::to_string( i ); } );
//  }
//}

/* FINISH DOCS! */
