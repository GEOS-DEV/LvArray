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
 * @file numpyHelpers.hpp
 * @brief Contains methods to help with conversion to python objects.
 */

#pragma once

// Source include
#include "pythonForwardDeclarations.hpp"
#include "pythonHelpers.hpp"
#include "../limits.hpp"

// system includes
#include <vector>
#include <typeindex>
#include <type_traits>

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 * @brief True if @c T can be represented by a NumPy native data type.
 * @tparam T The type to query.
 */
template< typename T >
constexpr bool canExportToNumpy = std::is_arithmetic< T >::value;

/**
 * @brief Return a NumPy ndarray that is a view into @c data.
 * @param data A pointer to the data to export.
 * @param type The type of the data to export.
 * @param dataIsConst True iff the data should be immutable. When this is true modifying the values
 *   in the returned ndarray raises an exception.
 * @param ndim The number of dimensions of @c data.
 * @param dims The size of each dimension.
 * @param strides The strides of each dimension.
 * @return A NumPy ndarray that is a view into @c data, or @c nullptr if there was an error.
 */
PyObject * createNumpyArrayImpl( void * const data,
                                 std::type_index const type,
                                 bool const dataIsConst,
                                 int const ndim,
                                 long long const * const dims,
                                 long long const * const strides );

/**
 * @brief Return a CuPy ndarray that is a view into @c data.
 * @param data A pointer to the data to export.
 * @param type The type of the data to export.
 * @param dataIsConst True iff the data should be immutable. Note that unlike NumPy this
 *   is not honored by CuPy, although this may change in the future.
 * @param ndim The number of dimensions of @c data.
 * @param dims The size of each dimension.
 * @param strides The strides of each dimension.
 * @return A CuPy ndarray that is a view into @c data, or @c nullptr if there was an error.
 */
PyObject * createCupyArrayImpl( void * const data,
                                std::type_index const type,
                                bool const dataIsConst,
                                int const ndim,
                                long long const * const dims,
                                long long const * const strides );

} // namespace internal

/**
 * @brief Attempt to import the NumPy API if it was not already imported, return @c true iff successful.
 * @return @c true iff the NumPy API was successfully imported.
 */
bool import_array_wrapper();

/**
 * @brief Return a NumPy ndarray view of @c data.
 * @tparam T The type of values in the array.
 * @tparam INDEX_TYPE The index type of the array.
 * @param data A pointer to the data to export.
 * @param modify If the ndarray should have modifiable values, only has an effect iff @tparam T is not @c const.
 * @param ndim The number of dimensions of @c data.
 * @param dimsPtr The size of each dimension.
 * @param stridesPtr The strides of each dimension.
 * @return A NumPy ndarray view of @c data.
 */
template< typename T, typename INDEX_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
createNumPyArray( T * const data,
                  bool const modify,
                  int const ndim,
                  INDEX_TYPE const * const dimsPtr,
                  INDEX_TYPE const * const stridesPtr )
{
  std::vector< long long > dims( ndim );
  std::vector< long long > strides( ndim );

  for( int i = 0; i < ndim; ++i )
  {
    dims[ i ] = integerConversion< long long >( dimsPtr[ i ] );
    strides[ i ] = integerConversion< long long >( stridesPtr[ i ] );
  }

  if( false )
  {
    return internal::createCupyArrayImpl( const_cast< void * >( static_cast< void const * >( data ) ),
                                          std::type_index( typeid( T ) ),
                                          std::is_const< T >::value || !modify,
                                          ndim,
                                          dims.data(),
                                          strides.data() );
  }
  return internal::createNumpyArrayImpl( const_cast< void * >( static_cast< void const * >( data ) ),
                                         std::type_index( typeid( T ) ),
                                         std::is_const< T >::value || !modify,
                                         ndim,
                                         dims.data(),
                                         strides.data() );
}

/**
 * @brief Create a NumPy 1D array of length 1 containing the scalar @c value.
 * @tparam T The type of the scalar @c value, if @c const then the returned array is immutable.
 * @param value The type to export.
 * @return A NumPy 1D array of length 1.
 */
template< typename T >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( T & value )
{
  long long dims = 1;
  long long strides = 1;

  return internal::createNumpyArrayImpl( const_cast< void * >( static_cast< void const * >( &value ) ),
                                         std::type_index( typeid( T ) ),
                                         std::is_const< T >::value,
                                         1,
                                         &dims,
                                         &strides );
}

/**
 * @brief Attempt to parse @c obj into a NumPy ndarray of type @c expectedType.
 * @param obj The object to parse.
 * @param expectedType The expected type of the values in the ndarray.
 * @return A tripple containing a PyArrayObject, a pointer to the data, and the size of the array.
 *   If there was an error then the returned value is @code { nullptr, nullptr, 0 } @endcode.
 */
std::tuple< PyObjectRef< PyObject >, void const *, long long >
parseNumPyArray( PyObject * const obj, std::type_index const expectedType );

/**
 * @brief Return the @c std::type_index corresponding to the NumPy type @c numpyType.
 * @param numpyType The NumPy type to convert.
 * @return The @c std::type_index corresponding to the NumPy type @c numpyType.
 */
std::type_index getTypeIndexFromNumPy( int const numpyType );

/**
 * @brief Return the name corresponding to the NumPy type @c numpyType.
 * @param numpyType The NumPy type to convert.
 * @return The name corresponding to the NumPy type @c numpyType.
 */
std::string getNumPyTypeName( int const numpyType );

/**
 * @brief Return the NumPy type object corresponding to @c typeIndex.
 * @param typeIndex The type to convert to a NumPy type.
 * @return The NumPy type object corresponding to @c typeIndex. If an error occurs a Python exception is set
 *   and a @c nullptr is returned.
 */
PyObject * getNumPyTypeObject( std::type_index const typeIndex );

} // namespace python
} // namespace LvArray
