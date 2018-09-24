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
#include "SortedArray.hpp"
#include <vector>
#include <string>

using size_type = SortedArray< int >::size_type;

namespace internal
{

/**
 * @brief Check that the SortedArray is equivalent to the std::vector. Checks equality using the
 * operator[], the iterator interface, and the raw pointer.
 * @param [in] v the SortedArray to check.
 * @param [in] v_ref the std::vector to check against. 
 */
template < class T >
void compare_to_reference( const SortedArray< T >& v, const std::set< T >& v_ref )
{
  ASSERT_EQ( v.size(), v_ref.size() );
  ASSERT_EQ( v.empty(), v_ref.empty() );
  if ( v.empty() )
  {
    ASSERT_EQ( v.size(), 0 );
    return;
  }

  typename SortedArray< T >::const_iterator it = v.begin();
  typename std::set< T >::const_iterator ref_it = v_ref.begin();
  size_type i = 0;
  const T* v_ptr = v.data();
  for ( ; it != v.end(); ++it )
  {
    ASSERT_EQ( v[ i ], *ref_it );
    ASSERT_EQ( *it, *ref_it );
    ASSERT_EQ( v_ptr[ i ], *ref_it );
    ++ref_it;
    ++i;
  }

  ASSERT_EQ( i, v.size() );
}

/**
 * @brief Test the push_back method of the ChaiVector.
 * @param [in/out] v the ChaiVector to check.
 * @param [in] n the number of values to append.
 * @param [in] get_value a function to generate the values to append.
 * @return the std::vector compared against.
 */
template < class T, class LAMBDA >
std::set< T > insert_test( SortedArray< T >& v, int n, LAMBDA get_value )
{
  EXPECT_TRUE( v.empty() );

  std::set< T > v_ref;
  for ( int i = 0; i < n; ++i )
  {
    const T& val = get_value( i );
    v.insert( val );
    v_ref.insert( val );
  }

  compare_to_reference( v, v_ref );
  return v_ref;
}

} /* namespace internal */


TEST( SortedArray, insert )
{
  constexpr int N = 1000;

  {
    SortedArray< int > v;
    internal::insert_test( v, N, []( int i ) -> int { return 2503 * i % 5857; } );
  }
}
