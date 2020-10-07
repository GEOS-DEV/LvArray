/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file carrayOutput.hpp
 * @brief Contains functions for outputting c-array objects.
 * @note These are in the global namespace.
 */

#pragma once

/**
 * @brief Output a c-array to a stream.
 * @tparam T The type contained in the array.
 * @tparam N The size of the array.
 * @param stream The output stream to write to.
 * @param array The c-array to output.
 * @return @p stream.
 */
template< typename T, int N >
std::enable_if_t< !std::is_same< T, char >::value, std::ostream & >
operator<< ( std::ostream & stream, T const ( &array )[ N ] )
{
  stream << "{ " << array[ 0 ];
  for( int i = 1; i < N; ++i )
  {
    stream << ", " << array[ i ];
  }
  stream << " }";
  return stream;
}

/**
 * @brief Output a 2D c-array to a stream.
 * @tparam T The type contained in the array.
 * @tparam M The size of the first dimension.
 * @tparam N The size of the second dimension.
 * @param stream The output stream to write to.
 * @param array The 2D c-array to output.
 * @return @p stream.
 */
template< typename T, int M, int N >
std::ostream & operator<< ( std::ostream & stream, T const ( &array )[ M ][ N ] )
{
  stream << "{ " << array[ 0 ];
  for( int i = 1; i < M; ++i )
  {
    stream << ", " << array[ i ];
  }
  stream << " }";
  return stream;
}
