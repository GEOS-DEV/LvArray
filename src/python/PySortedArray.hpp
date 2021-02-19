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

/**
 * @file
 */

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "numpyHelpers.hpp"
#include "../SortedArray.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"
#include "../output.hpp"

// System includes
#include <string>
#include <memory>
#include <typeindex>

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 * @class PySortedArrayWrapperBase
 * @brief Provides a virtual Python wrapper around a SortedArray.
 */
class PySortedArrayWrapperBase
{
public:

  /**
   * @brief Default destructor.
   */
  virtual ~PySortedArrayWrapperBase() = default;

  /**
   * @brief Return the access level for the SortedArray.
   * @return The access level for the SortedArray.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the SortedArray.
   * @param accessLevel The access level.
   * @param memorySpace The memory space to access in.
   */
  virtual void setAccessLevel( int const accessLevel, int const memorySpace ) = 0;

  /**
   * @brief Return a string representing the underlying SortedArray type.
   * @return A string representing the underlying SortedArray type.
   */
  virtual std::string repr() const = 0;

  /**
   * @brief Return the type of the values stored in the sub-arrays.
   * @return The type of the values stored in the sub-arrays.
   */
  virtual std::type_index valueType() const = 0;

  /**
   * @brief Insert values into the set.
   * @param values The values to insert. Must be of type @c valueType, of length @c nVals and sorted unique.
   * @param nVals The number of values to insert.
   * @return The number of values inserted.
   * @note This wraps SortedArray::insert.
   */
  virtual long long insert( void const * const values, long long const nVals ) = 0;

  /**
   * @brief Remove values from the set.
   * @param values The values to remove. Must be of type @c valueType, of length @c nVals and sorted unique.
   * @param nVals The number of values to remove.
   * @return The number of values removed.
   * @note This wraps SortedArray::remove.
   */
  virtual long long remove( void const * const values, long long const nVals ) = 0;

  /**
   * @brief Return an immutable NumPy array representing the set.
   * @return An immutable NumPy Array representing the set.
   */
  virtual PyObject * toNumPy() = 0;

protected:

  /**
   * @brief Construct an empty PySortedArrayWrapperBase.
   */
  PySortedArrayWrapperBase( ):
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /// The access level for the SortedArray.
  int m_accessLevel;
};

/**
 * @class PySortedArrayWrapper
 * @brief Provides a concrete implementation of PySortedArrayWrapperBase.
 * @tparam T The type of values in the SortedArray.
 * @tparam INDEX_TYPE The index type of the SortedArray.
 * @tparam BUFFER_TYPE The buffer type of the SortedArray.
 * @note This holds a reference to the wrapped SortedArray, you must ensure that the reference remains valid
 *   for the lifetime of this object.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PySortedArrayWrapper final : public PySortedArrayWrapperBase
{
public:

  /**
   * @brief Construct a new Python wrapper around @c sortedArray.
   * @param sortedArray The SortedArray to wrap.
   */
  PySortedArrayWrapper( SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & sortedArray ):
    PySortedArrayWrapperBase( ),
    m_sortedArray( sortedArray )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PySortedArrayWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int const accessLevel, int const memorySpace ) final override
  {
    LVARRAY_UNUSED_VARIABLE( memorySpace );
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< SortedArray< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const final override
  { return std::type_index( typeid( T ) ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long insert( void const * const values, long long const nVals ) final override
  {
    T const * const castedValues = reinterpret_cast< T const * >( values );
    return integerConversion< long long >( m_sortedArray.insert( castedValues, castedValues + nVals ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long remove( void const * const values, long long const nVals ) final override
  {
    T const * const castedValues = reinterpret_cast< T const * >( values );
    return integerConversion< long long >( m_sortedArray.remove( castedValues, castedValues + nVals ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * toNumPy() final override
  {
    INDEX_TYPE const dims = m_sortedArray.size();
    INDEX_TYPE const strides = 1;
    return createNumPyArray( m_sortedArray.data(), false, 1, &dims, &strides );
  }

private:
  /// The wrapped SortedArray.
  SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & m_sortedArray;
};

/**
 * @brief Create a Python object corresponding to @c sortedArray.
 * @param sortedArray The SortedArray to export to Python.
 * @return A Python object corresponding to @c sortedArray.
 */
PyObject * create( std::unique_ptr< internal::PySortedArrayWrapperBase > && sortedArray );

} // namespace internal

/**
 * @brief Create a Python object corresponding to @c sortedArray.
 * @tparam T The type of values in the SortedArray.
 * @tparam INDEX_TYPE The index type of the SortedArray.
 * @tparam BUFFER_TYPE The buffer type of the SortedArray.
 * @param sortedArray The SortedArray to export to Python.
 * @return A Python object corresponding to @c sortedArray.
 * @note The returned Python object holds a reference to @c sortedArray, you must ensure
 *   the reference remains valid throughout the lifetime of the object.
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & sortedArray )
{
  auto tmp = std::make_unique< internal::PySortedArrayWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( sortedArray );
  return internal::create( std::move( tmp ) );
}

/**
 * @brief Return the Python type for the SortedArray.
 * @return The Python type for the SortedArray.
 */
PyTypeObject * getPySortedArrayType();

} // namespace python
} // namespace LvArray
