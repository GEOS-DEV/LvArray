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

#ifndef CHAI_BUFFER_HPP
#define CHAI_BUFFER_HPP

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "Macros.hpp"
#include "Logger.hpp"
#include "arrayManipulation.hpp"
#include "StringUtilities.hpp"

// TPL includes
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"

// System includes
#include <mutex>


namespace LvArray
{

namespace internal
{
static std::mutex chaiLock;

inline std::string calculateSize( size_t const bytes )
{
  if (bytes >> 20 != 0)
  {
    return std::to_string(bytes >> 20) + "MB";
  }
  else
  {
    return std::to_string(bytes >> 10) + "KB";
  }
}

} // namespace internal

template < typename T >
class ChaiBuffer : public chai::CHAICopyable
{
public:
  using value_type = T;

  LVARRAY_HOST_DEVICE
  ChaiBuffer():
    m_array()
  {
#if !defined(__CUDA_ARCH__)
    setUserCallBack( "NoNameProvided" );
#endif
  }

  ChaiBuffer( std::ptrdiff_t const capacity ):
    ChaiBuffer()
  {
    std::lock_guard< std::mutex > lock( internal::chaiLock );
    m_array.allocate( capacity );
  }

  LVARRAY_HOST_DEVICE
  ChaiBuffer( std::nullptr_t ) :
    m_array( nullptr )
  {}

  LVARRAY_HOST_DEVICE inline
  ChaiBuffer & operator=( std::ptrdiff_t )
  {
    m_array = nullptr;
    return *this;
  }

  void reallocate( std::ptrdiff_t const size, std::ptrdiff_t const newCapacity )
  {
    internal::chaiLock.lock();
    chai::ManagedArray< T > newArray( newCapacity );
    internal::chaiLock.unlock();

    newArray.setUserCallback( m_array.getUserCallback() );

    arrayManipulation::moveInto( newBuf.data(), newCapacity, data(), size );

    free();
    *this = std::move( newBuf );
    registerTouch( chai::CPU );
  }

  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::ptrdiff_t capacity() const
  {
    return m_array.size();
  }

  inline
  void free()
  {
    std::lock_guard< std::mutex > lock( internal::chaiLock );
    m_array.free();
    m_array = nullptr;
  }
  
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  {
    return &m_array[0];
  }

  void move( chai::ExecutionSpace const space, bool const touch )
  {
#if defined(USE_CUDA)
    if ( capacity() == 0 ) return;

    void * ptr = const_cast< typename std::remove_const<T>::type * >( data() );
    if ( space == chai::ArrayManager::getInstance()->getPointerRecord( ptr )->m_last_space ) return;

    if ( touch ) m_array.move( space );
    else reinterpret_cast< chai::ManagedArray< T const > & >( m_array ).move( space );
#else
    CXX_UTILS_UNUSED_VARIABLE( space );
    CXX_UTILS_UNUSED_VARIABLE( touch );
#endif
  }

  void registerTouch( chai::ExecutionSpace const space )
  {
    m_array.registerTouch( space );
  }

  template < typename U = ChaiBuffer< T > >
  void setUserCallBack( std::string const & name )
  {
#if defined(USE_CUDA)
    std::string const typeString = cxx_utilities::demangle( typeid( U ).name() );
    m_array.setUserCallback( [name, typeString]( chai::Action act, chai::ExecutionSpace s, size_t bytes )
    {
      if (act == chai::ACTION_MOVE)
      {
        std::string const & size = internal::calculateSize( bytes );
        if (s == chai::CPU)
        {
          GEOS_LOG_RANK("Moved " << size << " to the CPU: " << typeString << " " << name );
        }
        else if (s == chai::GPU)
        {
          GEOS_LOG_RANK("Moved " << size << " to the GPU: " << typeString << " " << name );
        }
      }
    });
#else
    CXX_UTILS_UNUSED_VARIABLE( name );
#endif
  }

private:
  chai::ManagedArray< T > m_array;
};

} /* namespace LvArray */

#endif /* CHAI_BUFFER_HPP */
