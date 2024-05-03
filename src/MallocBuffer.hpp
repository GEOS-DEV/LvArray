/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file MallocBuffer.hpp
 * @brief Contains the the implementation of LvArray::MallocBuffer.
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"
#include "bufferManipulation.hpp"
#include "math.hpp"

// System includes
#include <stddef.h>

namespace LvArray
{

/**
 * @class MallocBuffer
 * @brief Implements the Buffer interface using malloc and free.
 * @tparam T type of data that is contained in the buffer.
 * @details Both the copy constructor and copy assignment constructor perform a shallow copy
 *   of the source. Similarly the destructor does not free the allocation.
 * @note The parent class bufferManipulation::VoidBuffer provides the default execution space related methods.
 */
template< typename T >
class MallocBuffer : public bufferManipulation::VoidBuffer
{
public:

  /// Alias used in the bufferManipulation functions.
  using value_type = T;

  /// Signifies that the MallocBuffer's copy semantics are shallow.
  static constexpr bool hasShallowCopy = true;

  /**
   * @brief Constructor for creating an empty or uninitialized buffer.
   * @note An uninitialized MallocBuffer is equivalent to an empty MallocBuffer and does
   *   not need to be free'd.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer( bool=true ):
    m_data( nullptr ),
    m_capacity( 0 )
  {}

  /**
   * @brief Copy constructor, creates a shallow copy.
   */
  MallocBuffer( MallocBuffer const & ) = default;

  /**
   * @brief Sized copy constructor, creates a shallow copy.
   * @param src The buffer to be coppied.
   */
  MallocBuffer( MallocBuffer const & src, std::ptrdiff_t ):
    MallocBuffer( src )
  {}

  /**
   * @brief Move constructor, creates a shallow copy.
   * @param src The buffer to be moved from, is empty after the move.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer( MallocBuffer && src ):
    m_data( src.m_data ),
    m_capacity( src.m_capacity )
  {
    src.m_capacity = 0;
    src.m_data = nullptr;
  }

  /**
   * @brief Create a shallow copy of @p src but with a different type.
   * @tparam U The type to convert from.
   * @param src The buffer to copy.
   */
  template< typename U >
  LVARRAY_HOST_DEVICE inline constexpr
  MallocBuffer( MallocBuffer< U > const & src ):
    m_data( reinterpret_cast< T * >( src.data() ) ),
    m_capacity( typeManipulation::convertSize< T, U >( src.capacity() ) )
  {}

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param src The buffer to be copied.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline LVARRAY_INTEL_CONSTEXPR
  MallocBuffer & operator=( MallocBuffer const & src )
  {
    m_capacity = src.m_capacity;
    m_data = src.m_data;
    return *this;
  }

  /**
   * @brief Move assignment operator, creates a shallow copy.
   * @param src The buffer to be moved from, is empty after the move.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE inline LVARRAY_INTEL_CONSTEXPR
  MallocBuffer & operator=( MallocBuffer && src )
  {
    m_capacity = src.m_capacity;
    m_data = src.m_data;
    src.m_capacity = 0;
    src.m_data = nullptr;
    return *this;
  }

  /**
   * @brief Reallocate the buffer to the new capacity.
   * @param size The number of values that are initialized in the buffer.
   * @param space The space to perform the reallocation in, not used.
   * @param newCapacity The new capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline
  void reallocate( std::ptrdiff_t const size, MemorySpace const space, std::ptrdiff_t const newCapacity )
  {
    LVARRAY_ERROR_IF_NE( space, MemorySpace::host );
    std::size_t newSpaceSize = newCapacity * sizeof( T );

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#endif
    // TODO: If std::is_trivially_copyable_v< T > then we could use std::realloc.
    T * const newPtr = reinterpret_cast< T * >( std::malloc( newSpaceSize ) );
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

    std::ptrdiff_t const overlapAmount = math::min( newCapacity, size );
    arrayManipulation::uninitializedMove( newPtr, overlapAmount, m_data );
    arrayManipulation::destroy( m_data, size );

    std::free( m_data );
    m_capacity = newCapacity;
    m_data = newPtr;
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values.
   * @note To destroy the values and free the data call bufferManipulation::free.
   */
  LVARRAY_HOST_DEVICE inline
  void free()
  {
    std::free( m_data );
    m_capacity = 0;
    m_data = nullptr;
  }

  /**
   * @return Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline
  std::ptrdiff_t capacity() const
  { return m_capacity; }

  /**
   * @return Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_data; }

  /**
   * @tparam INDEX_TYPE the type used to index into the values.
   * @return The value at position @p i .
   * @param i The position of the value to access.
   * @note No bounds checks are performed.
   */
  template< typename INDEX_TYPE >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator[]( INDEX_TYPE const i ) const
  { return m_data[ i ]; }

private:

  /// A pointer to the data.
  T * LVARRAY_RESTRICT m_data = nullptr;

  /// The size of the allocation.
  std::ptrdiff_t m_capacity = 0;
};

} // namespace LvArray
