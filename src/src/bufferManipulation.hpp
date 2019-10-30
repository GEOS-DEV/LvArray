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

#ifndef BUFFER_MANIPULATION_HPP_
#define BUFFER_MANIPULATION_HPP_

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "Logger.hpp"
#include "arrayManipulation.hpp"

// TPL includes
#include <chai/ManagedArray.hpp>
#include <chai/ArrayManager.hpp>

// System includes
#include <utility>

namespace bufferManipulation
{

  struct VoidBuffer
  {
    void move( chai::ExecutionSpace const CXX_UTILS_UNUSED_ARG(space),
             bool const CXX_UTILS_UNUSED_ARG(touch) )
    {}

    void registerTouch( chai::ExecutionSpace const CXX_UTILS_UNUSED_ARG(space) )
    {}

    template < typename >
    void setUserCallBack( std::string const & CXX_UTILS_UNUSED_ARG(name) )
    {}
  };

  template< typename BUFFER >
  inline
  void check( BUFFER const & buf, std::ptrdiff_t const size )
  { 
#ifdef USE_ARRAY_BOUNDS_CHECK
    if ( size > buf.capacity() )
    {
      GEOS_LOG("ERROR");
    }

    GEOS_ERROR_IF_GT( 0, buf.capacity() );
    GEOS_ERROR_IF_GT( 0, size );
    GEOS_ERROR_IF_GT( size, buf.capacity() );
#else
    CXX_UTILS_DEBUG_VAR( buf );
    CXX_UTILS_DEBUG_VAR( size );
#endif
  }

  template< typename BUFFER >
  inline
  void checkInsert( BUFFER const & buf, std::ptrdiff_t const size, std::ptrdiff_t const pos )
  {
#ifdef USE_ARRAY_BOUNDS_CHECK
    check( buf, size );
    GEOS_ERROR_IF_GT( 0, pos );
    GEOS_ERROR_IF_GT( pos, size );
#else
    CXX_UTILS_DEBUG_VAR( buf );
    CXX_UTILS_DEBUG_VAR( size );
    CXX_UTILS_DEBUG_VAR( pos );
#endif
  }

  template< typename BUFFER >
  void free( BUFFER & buf, std::ptrdiff_t const size )
  {
    check( buf, size );

    using T = typename BUFFER::value_type;

    T * const restrict data = buf.data();
    for ( std::ptrdiff_t i = 0; i < size; ++i )
    {
      data[ i ].~T();
    }

    buf.free();
  }

  template< typename BUFFER >
  void setCapacity( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    check( buf, size );
    buf.reallocate( size, newCapacity );
  }

  template< typename BUFFER >
  void reserve( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    check( buf, size );

    if ( newCapacity > buf.capacity() )
    {
      setCapacity( buf, size, newCapacity );
    }
  }

  template< typename BUFFER >
  void dynamicReserve( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    check( buf, size );

    if ( newCapacity > buf.capacity() )
    {
      setCapacity( buf, size, 2 * newCapacity );
    }
  }

  template< typename BUFFER, typename... ARGS >
  void resize( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const newSize, ARGS && ...args )
  {
    check( buf, size );

    reserve( buf, size, newSize );

    arrayManipulation::resize( buf.data(), size, newSize, std::forward< ARGS >( args )... );

    if ( newSize > 0 )
    {
      buf.registerTouch( chai::CPU );
    }
  }

  template< typename BUFFER >
  void pushBack( BUFFER & buf, std::ptrdiff_t const size, typename BUFFER::value_type const & value )
  {
    check( buf, size );

    dynamicReserve( buf, size, size + 1 );
    arrayManipulation::append( buf.data(), size, value );
  }

  template< typename BUFFER >
  void pushBack( BUFFER & buf, std::ptrdiff_t const size, typename BUFFER::value_type && value )
  {
    check( buf, size );

    dynamicReserve( buf, size, size + 1 );
    arrayManipulation::append( buf.data(), size, std::move( value ) );
  }

  template< typename BUFFER >
  void popBack( BUFFER & buf, std::ptrdiff_t const size )
  {
    check( buf, size );
    arrayManipulation::popBack( buf.data(), size );
  }

  template< typename BUFFER >
  void insert( BUFFER & buf,
               std::ptrdiff_t const size,
               std::ptrdiff_t const pos,
               typename BUFFER::value_type const & value )
  {
    checkInsert( buf, size, pos );

    dynamicReserve( buf, size, size + 1 );
    arrayManipulation::insert( buf.data(), size, pos, value );
  }

  template< typename BUFFER >
  void insert( BUFFER & buf,
               std::ptrdiff_t const size,
               std::ptrdiff_t const pos,
               typename BUFFER::value_type && value )
  {
    checkInsert( buf, size, pos );

    dynamicReserve( buf, size, size + 1 );
    arrayManipulation::insert( buf.data(), size, pos, std::move( value ) );
  }

  template< typename BUFFER >
  void insert( BUFFER & buf,
               std::ptrdiff_t const size,
               std::ptrdiff_t const pos,
               typename BUFFER::value_type const * const values,
               std::ptrdiff_t nVals )
  {
    checkInsert( buf, size, pos );

    dynamicReserve( buf, size, size + nVals );
    arrayManipulation::insert( buf.data(), size, pos, values, nVals );
  }

  template< typename BUFFER >
  void erase( BUFFER & buf, std::ptrdiff_t const size, std::ptrdiff_t const pos )
  {
    check( buf, size );
    GEOS_ERROR_IF_GE( pos, size );

    arrayManipulation::erase( buf.data(), size, pos, std::ptrdiff_t( 1 ) );
  }

  template< typename DST_BUFFER, typename SRC_BUFFER >
  void copyInto( DST_BUFFER & dst, std::ptrdiff_t const dstSize, SRC_BUFFER const & src, std::ptrdiff_t const srcSize )
  {
    check( dst, dstSize );
    check( src, srcSize );

    resize( dst, dstSize, srcSize );

    using T = typename DST_BUFFER::value_type;
    T * const restrict dstData = dst.data();
    T const * const restrict srcData = src.data();

    for ( std::ptrdiff_t i = 0; i < srcSize; ++i )
    {
      dstData[ i ] = srcData[ i ];
    }
  }


} // namespace bufferManipulations



#endif /* BUFFER_MANIPULATION_HPP_ */
