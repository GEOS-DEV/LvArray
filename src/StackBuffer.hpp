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

#ifndef STACKBUFER_HPP_
#define STACKBUFER_HPP_

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
   * @brief Notionally this method reallocates the buffer, but since the StackBuffer is sized at
   *   compile time all this does is check that newCapacity doesn't exceed LENGTH.
   * @param size The current size of the buffer, not used.
   * @param newCapacity the new capacity of the buffer.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    LVARRAY_UNUSED_VARIABLE( size );
    LVARRAY_ERROR_IF_GT( newCapacity, LENGTH );
  }
  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note For this class this is a no-op since T must be trivially destructable.
   */
  void free()
  {}

  /**
   * @brief @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  std::ptrdiff_t capacity() const
  { return LENGTH; }

  /**
   * @brief @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return const_cast< T * >( m_data ); }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @brief @return The value at position @p i .
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

#endif // STACKBUFER_HPP_
