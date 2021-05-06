/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file StackBuffer.hpp
 * @brief Contains the implementation of LvArray:StackBuffer.
 */


#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"
#include "bufferManipulation.hpp"

// System includes
#include <stddef.h>
#include <type_traits>


namespace LvArray
{

/**
 * @class StackBuffer
 * @brief This class implements the Buffer interface using a c-array.
 * @tparam T type of data that is contained in the buffer. T must be both
 *   trivially copyable and trivially destructable.
 * @tparam LENGTH the length of the buffer.
 * @note Unlike the standard Buffer classes the StackBuffer does not permit
 *   making shallow copies.
 * @note The parent class provides the default execution space related methods.
 */
template< typename T, int LENGTH >
class StackBuffer : public bufferManipulation::VoidBuffer
{
public:
  static_assert( std::is_trivially_destructible< T >::value, "The StackBuffer can only hold trivially copyable and destructable types." );

  /// Alias used in the bufferManipulation functions.
  using value_type = T;

  /// Signifies that the StackBuffer's copy semantics are deep.
  constexpr static bool hasShallowCopy = false;

  /**
   * @brief Constructor for creating an empty/uninitialized buffer.
   * @note For the StackBuffer an uninitialized buffer is equivalent to an empty buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  StackBuffer( bool=true )
  {}

  /**
   * @brief Sized copy constructor, creates a deep copy.
   * @param src The buffer to be coppied.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  StackBuffer( StackBuffer const & src, std::ptrdiff_t ):
    StackBuffer( src )
  {}

  /**
   * @brief Create a copy of @p src with const T.
   * @tparam _T A dummy parameter to allow enable_if, do not specify.
   * @param src The buffer to copy.
   */
  template< typename _T=T, typename=std::enable_if_t< std::is_const< _T >::value > >
  LVARRAY_HOST_DEVICE inline constexpr
  StackBuffer( StackBuffer< std::remove_const_t< T >, LENGTH > const & src ):
    StackBuffer( reinterpret_cast< StackBuffer const & >( src ) )
  {}

  /**
   * @brief Notionally this method reallocates the buffer, but since the StackBuffer is sized at
   *   compile time all this does is check that newCapacity doesn't exceed LENGTH.
   * @param size The current size of the buffer, not used.
   * @param space The space to perform the reallocation in, not used.
   * @param newCapacity the new capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline
  void reallocate( std::ptrdiff_t const size, MemorySpace const space, std::ptrdiff_t const newCapacity )
  {
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_UNUSED_VARIABLE( space );
    LVARRAY_ERROR_IF_GT( newCapacity, LENGTH );
  }
  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note For this class this is a no-op since T must be trivially destructable.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  void free()
  {}

  /**
   * @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  std::ptrdiff_t capacity() const
  { return LENGTH; }

  /**
   * @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return const_cast< T * >( m_data ); }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @return The value at position @p i .
   * @param i The position of the value to access.
   * @note No bounds checks are performed.
   */
  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator[]( INDEX_TYPE const i ) const
  { return const_cast< T * >( m_data )[ i ]; }

private:

  /// The c-array containing the values.
  T m_data[ LENGTH ];
};

} // namespace LvArray
