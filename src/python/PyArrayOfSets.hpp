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
#include "../ArrayOfSets.hpp"
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
 * @class PyArrayOfSetsWrapperBase
 * @brief Provides a virtual Python wrapper around an ArrayOfSets.
 */
class PyArrayOfSetsWrapperBase
{
public:

  /**
   * brief Default destructor.
   */
  virtual ~PyArrayOfSetsWrapperBase() = default;

  /**
   * @brief Return the access level for the array.
   * @return The access level for the array.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the array.
   * @param accessLevel The access level.
   */
  virtual void setAccessLevel( int const accessLevel ) = 0;

  /**
   * @brief Return a string representing the underlying ArrayOfSetsType.
   * @return a string representing the underlying ArrayOfSetsType.
   */
  virtual std::string repr() const = 0;

  /**
   * @brief Return the size of the array (number of sets).
   * @return The size of the array (number of sets).
   * @note This wraps ArrayOfSets::size.
   */
  virtual long long size() const = 0;

  /**
   * @brief Return an immutable 1D NumPy array representing the set at index @c setIndex.
   * @param setIndex The set to access.
   * @return An immutable 1D NumPy array representing the set at index @c setIndex.
   * @note This wraps ArrayOfSets::operator[].
   */
  virtual PyObject * operator[]( long long const setIndex ) = 0;

  /**
   * @brief Return the type of the values stored in the sets.
   * @return The type of the values stored in the sets.
   */
  virtual std::type_index valueType() const = 0;

  /**
   * @brief Erase the set at index @c setIndex.
   * @param setIndex The set to erase.
   * @note This wraps ArrayOfSets::eraseSet.
   */
  virtual void eraseSet( long long const setIndex ) = 0;

  /**
   * @brief Insert a new set at index @c setIndex.
   * @param setIndex The index to insert the new set at.
   * @param capacity The capacity of the new set.
   * @note This wraps ArrayOfSets::insertSet.
   */
  virtual void insertSet( long long const setIndex, long long const capacity ) = 0;

  /**
   * @brief Remove values from the a set.
   * @param setIndex The set to remove from.
   * @param values The values to remove. Must be of type @c valueType, of length @c numVals, and sorted unique.
   * @param numVals The number of values to remove.
   * @return The number of values removed.
   * @note This wraps ArrayOfSets::removeFromSet.
   */
  virtual long long removeFromSet( long long const setIndex, void const * const values, long long const numVals ) = 0;

  /**
   * @brief Insert values into a set.
   * @param setIndex The set to insert into.
   * @param values The values to insert. Must be of type @c valueType, of length @c numVals, and sorted unique.
   * @param numVals The number of values to insert.
   * @return The number of values inserted.
   * @note This wraps ArrayOfSets::insertIntoSet.
   */
  virtual long long insertIntoSet( long long const setIndex, void const * const values, long long const numVals ) = 0;

protected:

  /**
   * @brief Construct an empty PyArrayOfSetsWrapperBase.
   */
  PyArrayOfSetsWrapperBase():
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /// access level for the ArrayOfSets.
  int m_accessLevel;
};

/**
 * @class PyArrayOfSetsWrapper
 * @brief Provides a concrete implementation of PyArrayOfSetsWrapperBase.
 * @tparam T The type of the entries in the ArrayOfSets.
 * @tparam INDEX_TYPE The index type of the ArrayOfSets.
 * @tparam BUFFER_TYPE The buffer type of the ArrayOfSets.
 * @note This holds a reference to the wrapped ArrayOfSets, you must ensure that the reference remains valid
 *   for the lifetime of this object.
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyArrayOfSetsWrapper final : public PyArrayOfSetsWrapperBase
{
public:

  /**
   * @brief Construct a new Python wrapper around @c arrayOfSets.
   * @param arrayOfSets The ArrayOfSets to wrap.
   */
  PyArrayOfSetsWrapper( ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfSets ):
    PyArrayOfSetsWrapperBase( ),
    m_arrayOfSets( arrayOfSets )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyArrayOfSetsWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int const accessLevel ) final override
  {
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long size() const final override
  { return integerConversion< long long >( m_arrayOfSets.size() ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * operator[]( long long const setIndex ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( setIndex );
    ArraySlice< T const, 1, 0, INDEX_TYPE > slice = m_arrayOfSets[ convertedIndex ];
    T const * data = slice;
    constexpr INDEX_TYPE strides = 1;
    INDEX_TYPE size = slice.size();
    return createNumPyArray( data, false, 1, &size, &strides );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const final override
  { return std::type_index( typeid( T ) ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void eraseSet( long long const setIndex ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( setIndex );
    m_arrayOfSets.eraseSet( convertedIndex );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void insertSet( long long const setIndex, long long const capacity ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( setIndex );
    INDEX_TYPE convertedCapacity = integerConversion< INDEX_TYPE >( capacity );
    m_arrayOfSets.insertSet( convertedIndex, convertedCapacity );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long removeFromSet( long long const setIndex,
                                   void const * const values,
                                   long long const numVals ) final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( setIndex );
    T const * begin = static_cast< T const * >( values );
    T const * end = begin + numVals;
    return integerConversion< long long >( m_arrayOfSets.removeFromSet( convertedIndex, begin, end ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual long long insertIntoSet( long long const setIndex,
                                   void const * const values,
                                   long long const numVals ) final override
  {
    INDEX_TYPE convertedSetIndex = integerConversion< INDEX_TYPE >( setIndex );
    T const * begin = static_cast< T const * >( values );
    T const * end = begin + numVals;
    return integerConversion< long long >( m_arrayOfSets.insertIntoSet( convertedSetIndex, begin, end ) );
  }

private:
  /// The wrapped ArrayOfSets.
  ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & m_arrayOfSets;
};

/**
 * @brief Create a Python object corresponding to @c arrayOfSets.
 * @param arrayOfSets The ArrayOfSets to export to Python.
 * @return A Python object corresponding to @c arrayOfSets.
 */
PyObject * create( std::unique_ptr< internal::PyArrayOfSetsWrapperBase > && arrayOfSets );

} // namespace internal


/**
 * @brief Create a Python object corresponding to @c arrayOfSets.
 * @tparam T The type of the entries in the ArrayOfSets.
 * @tparam INDEX_TYPE The index type of the ArrayOfSets.
 * @tparam BUFFER_TYPE The buffer type of the ArrayOfSets.
 * @param arrayOfSets The ArrayOfSets to export to Python.
 * @return A Python object corresponding to @c arrayOfSets.
 * @note The returned Python object holds a reference to @c arrayOfSets, you must ensure
 *   the reference remains valid throughout the lifetime of the object.
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfSets )
{
  auto tmp = std::make_unique< internal::PyArrayOfSetsWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( arrayOfSets );
  return internal::create( std::move( tmp ) );
}

/**
 * @brief Return the Python type for the ArrayOfSets.
 * @return The Python type object for the ArrayOfSets.
 */
PyTypeObject * getPyArrayOfSetsType();

} // namespace python
} // namespace LvArray
