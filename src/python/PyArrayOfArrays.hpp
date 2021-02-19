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
#include "../ArrayOfArrays.hpp"
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
 * @class PyArrayOfArraysWrapperBase
 * @brief Provides a virtual Python wrapper around an ArrayOfArrays.
 */
class PyArrayOfArraysWrapperBase
{
public:

  /**
   * @brief Default destructor.
   */
  virtual ~PyArrayOfArraysWrapperBase() = default;

  /**
   * @brief Return the access level for the ArrayOfArrays.
   * @return The access level for the ArrayOfArrays.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the ArrayOfArrays.
   * @param accessLevel The access level.
   */
  virtual void setAccessLevel( int const accessLevel ) = 0;

  /**
   * @brief Return a string representing the underlying ArrayOfArrays type.
   * @return a string representing the underlying ArrayOfArrays type.
   */
  virtual std::string repr() const = 0;

  /**
   * @brief Return the size of the array (number of sub-arrays).
   * @return The size of the array (number of sub-arrays).
   * @note This wraps ArrayOfArrays::size.
   */
  virtual long long size() const = 0;

  /**
   * @brief Return the size of the array at index @c arrayIndex.
   * @param arrayIndex The array to query.
   * @return The size of the array at index @c arrayIndex.
   * @note This wraps ArrayOfArrays::sizeOfArray.
   */
  virtual long long sizeOfArray( long long arrayIndex ) const = 0;

  /**
   * @brief Return a 1D numpy array representing the array at index @c arrayIndex.
   * @param arrayIndex The array to access.
   * @return An immutable 1D numpy array representing the array at index @c arrayIndex.
   * @note This wraps ArrayOfArrays::operator[].
   */
  virtual PyObject * operator[]( long long arrayIndex ) = 0;

  /**
   * @brief Return the type of the values stored in the sub-arrays.
   * @return The type of the values stored in the sub-arrays.
   */
  virtual std::type_index valueType() const = 0;

  /**
   * @brief Erase the array at index @c arrayIndex.
   * @param arrayIndex The array to erase.
   * @note This wraps ArrayOfArrays::eraseArray.
   */
  virtual void eraseArray( long long arrayIndex ) = 0;

  /**
   * @brief Erase a value from an array.
   * @param arrayIndex The index of the array to erase from.
   * @param valueIndex The index of the value to erase from the array.
   * @note This wraps ArrayOfArrays::eraseFromArray.
   */
  virtual void eraseFromArray( long long arrayIndex, long long valueIndex ) = 0;

  /**
   * @brief Insert a new array.
   * @param arrayIndex The index at which to insert the new array.
   * @param values The values of the array to insert, must be of type @c valueType() and length @c numVals.
   * @param numVals The number of values in the array to insert.
   * @note This wraps ArrayOfArrays::insertArray.
   */
  virtual void insertArray( long long arrayIndex, void const * values, long long numVals ) = 0;

  /**
   * @brief Insert new values into an existing array.
   * @param arrayIndex The index of the array to insert into.
   * @param valueIndex The index at which to insert into the array.
   * @param values The values to insert, must be of type @c valueType() and length @c numVals.
   * @param numVals The number of values to insert.
   * @note This wraps ArrayOfArrays::insertIntoArray.
   */
  virtual void insertIntoArray( long long arrayIndex,
                                long long valueIndex,
                                void const * values,
                                long long numVals ) = 0;

protected:

  /**
   * @brief Construct an empty PyArrayOfArraysWrapperBase.
   */
  PyArrayOfArraysWrapperBase():
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /// access level for the ArrayOfArrays.
  int m_accessLevel;
};

/**
 * @class PyArrayOfArraysWrapper
 * @brief Provides a concrete implementation of PyArrayOfArraysWrapperBase.
 * @tparam T The type of the entries in the ArrayOfArrays.
 * @tparam INDEX_TYPE The index type of the ArrayOfArrays.
 * @tparam BUFFER_TYPE The buffer type of the ArrayOfArrays.
 * @note This holds a reference to the wrapped ArrayOfArrays, you must ensure that the reference remains valid
 *   for the lifetime of this object.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyArrayOfArraysWrapper final : public PyArrayOfArraysWrapperBase
{
public:

  /**
   * @brief Construct a new Python wrapper around @c arrayOfArrays.
   * @param arrayOfArrays The ArrayOfArrays to wrap.
   */
  PyArrayOfArraysWrapper( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfArrays ):
    PyArrayOfArraysWrapperBase( ),
    m_arrayOfArrays( arrayOfArrays )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyArrayOfArraysWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int accessLevel ) final override
  {
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long size() const final override
  { return integerConversion< long long >( m_arrayOfArrays.size() ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long sizeOfArray( long long arrayIndex ) const final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( arrayIndex );
    return integerConversion< long long >( m_arrayOfArrays.sizeOfArray( convertedIndex ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * operator[]( long long arrayIndex ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( arrayIndex );
    ArraySlice< T, 1, 0, INDEX_TYPE > slice = m_arrayOfArrays[ convertedIndex ];
    T * data = slice;
    constexpr INDEX_TYPE strides = 1;
    INDEX_TYPE size = slice.size();
    return createNumPyArray( data, getAccessLevel() >= static_cast< int >( LvArray::python::PyModify::MODIFIABLE ), 1, &size, &strides );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const final override
  { return std::type_index( typeid( T ) ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void eraseArray( long long arrayIndex ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( arrayIndex );
    m_arrayOfArrays.eraseArray( convertedIndex );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void eraseFromArray( long long arrayIndex, long long valueIndex ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( arrayIndex );
    INDEX_TYPE convertedBegin = integerConversion< INDEX_TYPE >( valueIndex );
    m_arrayOfArrays.eraseFromArray( convertedIndex, convertedBegin );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void insertArray( long long arrayIndex, void const * values, long long numVals ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( arrayIndex );
    T const * begin = static_cast< T const * >( values );
    T const * end = begin + numVals;
    m_arrayOfArrays.insertArray( convertedIndex, begin, end );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void insertIntoArray( long long arrayIndex,
                                long long valueIndex,
                                void const * values,
                                long long numVals ) final override
  {
    INDEX_TYPE convertedArray = integerConversion< INDEX_TYPE >( arrayIndex );
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( valueIndex );
    T const * begin = static_cast< T const * >( values );
    T const * end = begin + numVals;
    m_arrayOfArrays.insertIntoArray( convertedArray, convertedIndex, begin, end );
  }

private:
  /// The wrapped ArrayOfArrays.
  ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & m_arrayOfArrays;
};

/**
 * @brief Create a Python object corresponding to @c arrayOfArrays.
 * @param arrayOfArrays The ArrayOfArrays to export to Python.
 * @return A Python object corresponding to @c arrayOfArrays.
 */
PyObject * create( std::unique_ptr< internal::PyArrayOfArraysWrapperBase > && arrayOfArrays );

} // namespace internal

/**
 * @brief Create a Python object corresponding to @c arrayOfArrays.
 * @tparam T The type of the entries in the ArrayOfArrays.
 * @tparam INDEX_TYPE The index type of the ArrayOfArrays.
 * @tparam BUFFER_TYPE The buffer type of the ArrayOfArrays.
 * @param arrayOfArrays The ArrayOfArrays to export to Python.
 * @return A Python object corresponding to @c arrayOfArrays.
 * @note The returned Python object holds a reference to @c arrayOfArrays, you must ensure
 *   the reference remains valid throughout the lifetime of the object.
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfArrays )
{
  auto tmp = std::make_unique< internal::PyArrayOfArraysWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( arrayOfArrays );
  return internal::create( std::move( tmp ) );
}

/**
 * @brief Return the Python type object for the ArrayOfArrays.
 * @return The Python type object for the ArrayOfArrays.
 */
PyTypeObject * getPyArrayOfArraysType();

} // namespace python
} // namespace LvArray
