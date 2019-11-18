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

#ifndef MALLOC_BUFFER_HPP_
#define MALLOC_BUFFER_HPP_

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "Logger.hpp"
#include "bufferManipulation.hpp"

// TPL includes
#include <stddef.h>


namespace LvArray
{

/**
 * @class MallocBuffer
 * @brief This class implements the Buffer interface using the standard library
 *        functions malloc and free. It also serves as a reference for the standard
 *        functionality of the Buffer classes.
 * @tparam T type of data that is contained in the buffer.
 * @note Both the copy constructor and copy assignment constructor perform a shallow copy
 *       of the source. Similarly the destructor does not free the allocation. This is
 *       the standard behavior of the Buffer classes.
 * @note The parent class provides the default execution space related methods.
 */
template< typename T >
class MallocBuffer : public bufferManipulation::VoidBuffer
{
public:

  // Alias used in the bufferManipulation functions.
  using value_type = T;

  /**
   * @brief Constructor for creating an empty/uninitialized buffer. For the MallocBuffer
   *        an uninitialized buffer is equivalent to an empty buffer.
   */
  MallocBuffer( bool=true ) :
    m_capacity( 0 ),
    m_data( nullptr )
  {}
  
  /**
   * @brief Reallocate the buffer to the new capacity.
   * @param size the number of values that are initialized in the buffer.
   *        values between [0, size) are destroyed.
   * @param newCapacity the new capacity of the buffer.
   */
  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    // If std::is_trivially_copyable_v< T > then we could use std::realloc.
    T * const newPtr = std::malloc( newCapacity * sizeof( T ) );
    arrayManipulation::moveInto( newPtr, newCapacity, m_data, size );

    std::free( m_data );
    m_capacity = newCapacity;
    m_data = newPtr;
  }
  
  /**
   * @brief Free the data in the buffer but does not destroy any values. To
   *        properly destroy the values and free the data call bufferManipulation::free.
   */
  void free()
  {
    std::free( m_data );
    m_capacity = 0;
    m_data = nullptr;
  }

  /**
   * @brief Return the capacity of the buffer.
   */
  inline CONSTEXPRFUNC
  std::ptrdiff_t capacity() const
  { return 0; }

  /**
   * @brief Return a pointer to the beginning of the buffer.
   */
  inline CONSTEXPRFUNC
  T * data() const
  {
    return m_data;
  }

private:
  std::ptrdiff_t m_capacity = 0;
  T * m_data = nullptr;
};

} // namespace LvArray

#endif // MALLOC_BUFFER_HPP_
