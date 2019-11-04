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
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "Logger.hpp"
#include "bufferManipulation.hpp"
#include "Array.hpp"

// TPL includes
#include <stddef.h>


namespace LvArray
{

/**
 * @class StackBuffer
 * @brief This class implements the Buffer interface using a stack array.
 * @tparam T type of data that is contained in the buffer.
 * @tparam LENGTH the length of the buffer.
 * @note Unlike the standard Buffer classes the StackBuffer does not permit
 *       making shallow copies.
 * @note The parent class provides the default execution space related methods.
 */
template< typename T, int LENGTH >
class StackBuffer : public bufferManipulation::VoidBuffer
{
public:
  using value_type = T;

  StackBuffer() = default;

  /**
   * @brief Constructor for creating an uninitialized Buffer. In general an
   *        uninitialized buffer is an undefined state and may only be assigned to
   *        however in this case it is the same as the default constructor.
   */
  StackBuffer( std::nullptr_t ) :
    StackBuffer()
  {}

  /**
   * @brief Reallocate the buffer to the new capacity. If the new capacity is
   *        greater than LENGTH this method will error out.
   * @param size the number of values that are initialized in the buffer.
   *        values between [0, size) are destroyed.
   * @param newCapacity the new capacity of the buffer.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    GEOS_ERROR_IF_GT( newCapacity, LENGTH );

    for( std::ptrdiff_t i = newCapacity ; i < size ; ++i )
    {
      m_data[ i ].~T();
    }
  }

  /**
   * @brief Free the data in the buffer but does not destroy any values.
   *        For this class this is a no-op.
   */
  void free()
  {}

  /**
   * @brief Return the capacity of the buffer.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::ptrdiff_t capacity() const
  {
    return LENGTH;
  }

  /**
   * @brief Return a pointer to the beginning of the buffer.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  {
    return const_cast< T * >( m_data );
  }

private:
  T m_data[ LENGTH ];
};

namespace internal
{

// Since Array expects the BUFFER to only except a single template parameter we need to
// create an alias for a StackBuffer with a given length.
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          int LENGTH > 
struct StackArrayHelper
{
  template< typename U >
  using BufferType = StackBuffer< U, LENGTH >;

  using type = Array< T, NDIM, PERMUTATION, INDEX_TYPE, BufferType >;
};

} // namespace internal

// An alias for a Array backed by a StackBuffer of a given length.
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          int LENGTH > 
using StackArray = typename internal::StackArrayHelper< T, NDIM, PERMUTATION, INDEX_TYPE, LENGTH >::type;

} // namespace LvArray

#endif // STACKBUFER_HPP_
