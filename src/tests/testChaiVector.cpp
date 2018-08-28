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
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "gtest/gtest.h"
#include "ChaiVector.hpp"
#include <vector>
#include <string>

using size_type = ChaiVector< int >::size_type;

namespace internal
{

/**
 * @brief Check that the ChaiVector is equivalent to the std::vector. Checks equality using the
 * operator[], the iterator interface, and the raw pointer.
 * @param [in] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to check against. 
 */
template < class T >
void compare_to_reference( const ChaiVector< T >& v, const std::vector< T >& v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if ( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  for ( size_type i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], v_ref[ i ] );
  }

  ASSERT_EQ( v.front(), v_ref.front() );
  ASSERT_EQ( v.back(), v_ref.back() );

  typename ChaiVector< T >::const_iterator it = v.begin();
  typename std::vector< T >::const_iterator ref_it = v_ref.begin();
  for ( ; it != v.end(); ++it )
  {
    ASSERT_EQ( *it, *ref_it );
    ++ref_it;
  }

  const T* v_ptr = v.data();
  const T* ref_ptr = v_ref.data();
  for ( size_type i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v_ptr[ i ], ref_ptr[ i ] );
  }
}

/**
 * @brief Test the push_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of values to append.
 * @param [in] get_value a function to generate the values to append.
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< T > push_back_test( ChaiVector< T >& v, int n, LAMBDA get_value )
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
 * @brief Test the insert method of the ChaiVector by inserting one value at a time.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of values to insert.
 * @param [in] get_value a function to generate the values to insert. 
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::vector< T > insert_test( ChaiVector< T >& v, int n, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );

  std::vector< T > v_ref;
  for ( int i = 0; i < n; ++i )
  {
    const T& val = get_value( i );
    if ( i % 3 == 0 )       /* Insert at the beginning. */
    {
      v.insert( v.begin(), val );
      v_ref.insert( v_ref.begin(), val );
    }
    else if ( i % 3 == 1 )  /* Insert at the end. */
    {
      v.insert( v.end(), val );
      v_ref.insert( v_ref.end(), val );
    }
    else                    /* Insert in the middle. */
    {
      v.insert( v.begin() + v.size() / 2, val );
      v_ref.insert( v_ref.begin() + v_ref.size() / 2, val );
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
std::vector< T > insert_multiple_test( ChaiVector< T >& v, int n, int m, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );
  
  std::vector< T > v_insert;
  std::vector< T > v_ref;
  for ( int i = 0; i < n; ++i )
  {
    v_insert.clear();
    for ( int j = 0; j < m; ++j )
    {
      v_insert.push_back( get_value( m * i + j ) );
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
 * @brief Test the erase method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] v_ref the std::vector to compare against.
 */
template < class T >
void erase_test( ChaiVector< T >& v, std::vector< T >& v_ref )
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
template < class T >
void pop_back_test( ChaiVector< T >& v, std::vector< T >& v_ref )
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
void resize_test( ChaiVector< T >& v, int n, LAMBDA get_value )
{
  ASSERT_TRUE( v.empty() );
  
  v.resize( n / 2 );

  ASSERT_EQ( v.size(), n / 2 );
  ASSERT_EQ( v.capacity(), n / 2 );

  T* data_ptr = v.data();
  for ( int i = 0; i < n / 2; ++i )
  {
    ASSERT_EQ( data_ptr[ i ], T() );
    data_ptr[ i ] = get_value( i );
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
    v[ i ] = get_value( 2 * i );
  }

  for ( int i = 0; i < n; ++i )
  {
    ASSERT_EQ( v[ i ], get_value( 2 * i ) );
  }
}

/**
 * @brief Test the reserve method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the end size of the vector.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void reserve_test( ChaiVector< T >& v, int n, LAMBDA get_value )
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
 * @brief Test the deep_copy method of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void deep_copy_test( const ChaiVector< T >& v, LAMBDA get_value )
{
  ChaiVector< T > v_cpy = v.deep_copy();
  
  ASSERT_EQ( v.size(), v_cpy.size() );
  ASSERT_EQ( v.capacity(), v_cpy.capacity() );

  ASSERT_NE( v.data(), v_cpy.data() );

  for ( size_type i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], v_cpy[ i ] );
    ASSERT_EQ( v[ i ], get_value( i ) );
  }

  for ( size_type i = 0; i < v.size(); ++i )
  {
    v_cpy[ i ] = get_value( 2 * i );
  }

  for ( size_type i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v_cpy[ i ], get_value( 2 * i ) );
    ASSERT_EQ( v[ i ], get_value( i ) );
  }
}

/**
 * @brief Test the shallow copy copy-constructor of the ChaiVector.
 * @param [in/out] v the ChaiVector to copy.
 * @param [in] get_value a function to generate the values. 
 */
template < class T, class LAMBDA >
void shallow_copy_test( const ChaiVector< T >& v, LAMBDA get_value )
{
  {
    ChaiVector< T > v_cpy(v);
    
    ASSERT_TRUE( v_cpy.isCopy() );
    ASSERT_EQ( v.size(), v_cpy.size() );
    ASSERT_EQ( v.capacity(), v_cpy.capacity() );

    ASSERT_EQ( v.data(), v_cpy.data() );

    for ( size_type i = 0; i < v.size(); ++i )
    {
      ASSERT_EQ( v[ i ], v_cpy[ i ] );
      ASSERT_EQ( v[ i ], get_value( i ) );
    }

    for ( size_type i = 0; i < v.size(); ++i )
    {
      v_cpy[ i ] = get_value( 2 * i );
    }
  }

  for ( size_type i = 0; i < v.size(); ++i )
  {
    ASSERT_EQ( v[ i ], get_value( 2 * i ) );
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


TEST( ChaiVector, push_back )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, insert )
{
  constexpr int N = 300;

  {
    ChaiVector< int > v;
    internal::insert_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::insert_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, insert_multiple )
{
  constexpr int N = 100;
  constexpr int M = 10;

  {
    ChaiVector< int > v;
    internal::insert_multiple_test( v, N, M, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::insert_multiple_test( v, N, M, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, erase )
{
  constexpr int N = 200;

  {
    ChaiVector< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::erase_test( v, v_ref );
  }

  {
    ChaiVector< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::erase_test( v, v_ref );
  }
}

TEST( ChaiVector, pop_back )
{
  constexpr int N = 300;

  {
    ChaiVector< int > v;
    std::vector< int > v_ref = internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::pop_back_test( v, v_ref );
  }

  {
    ChaiVector< Tensor > v;
    std::vector< Tensor > v_ref =  internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::pop_back_test( v, v_ref );
  }
}

TEST( ChaiVector, resize )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::resize_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::resize_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, reserve )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::reserve_test( v, N, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::reserve_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, deep_copy )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::deep_copy_test( v, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::deep_copy_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}

TEST( ChaiVector, shallow_copy )
{
  constexpr int N = 1000;

  {
    ChaiVector< int > v;
    internal::push_back_test( v, N, []( int i ) -> int { return i; } );
    internal::shallow_copy_test( v, []( int i ) -> int { return i; } );
  }

  {
    ChaiVector< Tensor > v;
    internal::push_back_test( v, N, []( int i ) -> Tensor { return Tensor( i ); } );
    internal::shallow_copy_test( v, []( int i ) -> Tensor { return Tensor( i ); } );
  }
}
