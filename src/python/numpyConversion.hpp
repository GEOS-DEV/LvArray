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
 * @file numpyConversion.hpp
 */

#pragma once
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL LvArray_ARRAY_API

// source includes
#include "../StringUtilities.hpp"
#include "../IntegerConversion.hpp"

// TPL includes (note python.h must be included before any stdlib headers)
#include <Python.h>
#include <numpy/arrayobject.h>

// system includes
#include <typeindex>
#include <type_traits>

namespace LvArray
{

namespace python
{

namespace internal
{

/**
 * @brief Get the Numpy type-flag corresponding to char.
 */
static constexpr int getCharFlag(void){
    if ( std::is_signed< char >::value )
        return NPY_BYTE;
    return NPY_UBYTE;
}

// Numpy flags for supported types
template< typename T >
static constexpr int numpyFlag = NPY_NOTYPE; // invalid type
template<>
static constexpr int numpyFlag< char > = getCharFlag();
template<>
static constexpr int numpyFlag< signed char > = NPY_BYTE;
template<>
static constexpr int numpyFlag< unsigned char > = NPY_UBYTE;
template<>
static constexpr int numpyFlag< short > = NPY_SHORT;
template<>
static constexpr int numpyFlag< unsigned short > = NPY_USHORT;
template<>
static constexpr int numpyFlag< int > = NPY_INT;
template<>
static constexpr int numpyFlag< unsigned int > = NPY_UINT;
template<>
static constexpr int numpyFlag< long > = NPY_LONG;
template<>
static constexpr int numpyFlag< unsigned long > = NPY_ULONG;
template<>
static constexpr int numpyFlag< long long > = NPY_LONGLONG;
template<>
static constexpr int numpyFlag< unsigned long long > = NPY_ULONGLONG;
template<>
static constexpr int numpyFlag< float > = NPY_FLOAT;
template<>
static constexpr int numpyFlag< double > = NPY_DOUBLE;
template<>
static constexpr int numpyFlag< long double > = NPY_LONGDOUBLE;

/**
 * @brief Create a numpy ndarray representing the given array. If T is const then the array should be immutable.
 * @tparam T the type of each entry in the array
 * @tparam INDEX_TYPE the integer type used to index the array, will be converted to npy_intp
 * @param data a pointer to the array.
 * @param ndim the number of dimensions in the array.
 * @param dims the number of elements in each dimension.
 * @param strides the element-wise strides to reach the next element in each dimension.
 */
template< typename T, typename INDEX_TYPE >
PyObject * create( T * const data, int ndim, INDEX_TYPE const * const dims, INDEX_TYPE const * const strides )
{
    using nonconstType = std::remove_const_t< T >;
    int flags = 0;
    static_assert( numpyFlag< nonconstType > != NPY_NOTYPE, "Unsupported type for conversion to numpy array");
    std::vector<npy_intp> byteStrides( ndim );
    std::vector<npy_intp> npyDims( ndim );
    nonconstType * npyData = const_cast< nonconstType* >( data );
    if ( !std::is_const< T >::value ){
        flags = NPY_ARRAY_WRITEABLE;
    }
    for ( int i = 0; i < ndim; ++i )
    {
        byteStrides[ i ] = integerConversion< npy_intp >( strides[ i ] * sizeof( T ) );
        npyDims[ i ] = integerConversion< npy_intp >( dims[ i ] );
    }
    return PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType( numpyFlag< nonconstType > ),
        ndim,
        npyDims.data(),
        byteStrides.data(),
        npyData,
        flags,
        NULL
    );
}

} // namespace internal

} // namespace python

} // namespace LvArray
