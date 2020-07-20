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
#include <climits>

namespace LvArray
{

namespace python
{

namespace internal
{

/**
 * @brief Get the Numpy type-flag corresponding to a `std::type_info` object.
 * @param tid the type_info object to use
 */
int getFlags( const std::type_info& tid ){
    if ( std::type_index( tid ) == std::type_index( typeid( char ) ) ){
        if ( CHAR_MIN < 0 )
            return NPY_BYTE;
        return NPY_UBYTE;
    }
    if ( std::type_index( tid ) == std::type_index( typeid( signed char ) ) )
        return NPY_BYTE;
    else if ( std::type_index( tid ) == std::type_index( typeid( unsigned char ) ) )
        return NPY_UBYTE;
    else if ( std::type_index( tid ) == std::type_index( typeid( short ) ) )
        return NPY_SHORT;
    else if ( std::type_index( tid ) == std::type_index( typeid( unsigned short ) ) )
        return NPY_USHORT;
    else if ( std::type_index( tid ) == std::type_index( typeid( int ) ) )
        return NPY_INT;
    else if ( std::type_index( tid ) == std::type_index( typeid( unsigned int ) ) )
        return NPY_UINT;
    else if ( std::type_index( tid ) == std::type_index( typeid( long ) ) )
        return NPY_LONG;
    else if ( std::type_index( tid ) == std::type_index( typeid( unsigned long ) ) )
        return NPY_ULONG;
    else if ( std::type_index( tid ) == std::type_index( typeid( long long ) ) )
        return NPY_LONGLONG;
    else if ( std::type_index( tid ) == std::type_index( typeid( unsigned long long ) ) )
        return NPY_ULONGLONG;
    else if ( std::type_index( tid ) == std::type_index( typeid( float ) ) )
        return NPY_FLOAT;
    else if ( std::type_index( tid ) == std::type_index( typeid( double ) ) )
        return NPY_DOUBLE;
    else if ( std::type_index( tid ) == std::type_index( typeid( long double ) ) )
        return NPY_LONGDOUBLE;
    else {
        LVARRAY_ERROR( "Unsupported type: " << demangle( tid.name() ) );
        return -1;
    }
}

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
        PyArray_DescrFromType( getFlags( typeid( T ) ) ),
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
