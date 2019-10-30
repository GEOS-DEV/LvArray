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

#ifndef STACKARRAYWRAPPER_HPP_
#define STACKARRAYWRAPPER_HPP_

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

template< typename T, int LENGTH >
class StackBuffer : public bufferManipulation::VoidBuffer
{
public:
  using value_type = T;

  StackBuffer() = default;

  StackBuffer( std::ptrdiff_t const initialCapacity )
  {
    GEOS_ERROR_IF_GT( initialCapacity, LENGTH );
  }

  StackBuffer( std::nullptr_t ) :
    StackBuffer()
  {}

  StackBuffer & operator=( std::nullptr_t )
  {
    return *this;
  }

  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    GEOS_ERROR_IF_GT( newCapacity, LENGTH );

    for( std::ptrdiff_t i = newCapacity ; i < size ; ++i )
    {
      m_data[ i ].~T();
    }
  }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::ptrdiff_t capacity() const
  {
    return LENGTH;
  }

  void free()
  {}

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

template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          int LENGTH > 
using StackArray = typename internal::StackArrayHelper< T, NDIM, PERMUTATION, INDEX_TYPE, LENGTH >::type;

} // namespace LvArray

#endif
