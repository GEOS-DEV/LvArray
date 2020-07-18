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

// Python must be the first include.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Source includes
#include "numpyHelpers.hpp"
#include "pythonHelpers.hpp"
#include "../system.hpp"

// System includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

#include <tuple>

namespace LvArray
{
namespace python
{

namespace internal
{

/**
 * @brief Return the NumPy type and bytesize of the type @c typeIndex.
 * @param typeIndex The type to convert.
 * @return A pair containing the NumPy type and the bytesize of @c typeIndex.
 *   If the conversion fails the returned NumPy type is @c NPY_NOTYPE.
 */
static std::pair< int, std::size_t > getNumPyType( std::type_index const typeIndex )
{
  if( typeIndex == std::type_index( typeid( char ) ) )
  {
    return { std::is_signed< char >::value ? NPY_BYTE : NPY_UBYTE, sizeof( char ) };
  }
  if( typeIndex == std::type_index( typeid( signed char ) ) )
  {
    return { NPY_BYTE, sizeof( signed char ) };
  }
  if( typeIndex == std::type_index( typeid( unsigned char ) ) )
  {
    return { NPY_UBYTE, sizeof( unsigned char ) };
  }
  if( typeIndex == std::type_index( typeid( short ) ) )
  {
    return { NPY_SHORT, sizeof( short ) };
  }
  if( typeIndex == std::type_index( typeid( unsigned short ) ) )
  {
    return { NPY_USHORT, sizeof( unsigned short ) };
  }
  if( typeIndex == std::type_index( typeid( int ) ) )
  {
    return { NPY_INT, sizeof( int ) };
  }
  if( typeIndex == std::type_index( typeid( unsigned int ) ) )
  {
    return { NPY_UINT, sizeof( unsigned int ) };
  }
  if( typeIndex == std::type_index( typeid( long ) ) )
  {
    return { NPY_LONG, sizeof( long ) };
  }
  if( typeIndex == std::type_index( typeid( unsigned long ) ) )
  {
    return { NPY_ULONG, sizeof( unsigned long ) };
  }
  if( typeIndex == std::type_index( typeid( long long ) ) )
  {
    return { NPY_LONGLONG, sizeof( long long ) };
  }
  if( typeIndex == std::type_index( typeid( unsigned long long ) ) )
  {
    return { NPY_ULONGLONG, sizeof( unsigned long long ) };
  }
  if( typeIndex == std::type_index( typeid( float ) ) )
  {
    return { NPY_FLOAT, sizeof( float ) };
  }
  if( typeIndex == std::type_index( typeid( double ) ) )
  {
    return { NPY_DOUBLE, sizeof( double ) };
  }
  if( typeIndex == std::type_index( typeid( long double ) ) )
  {
    return { NPY_LONGDOUBLE, sizeof( long double ) };
  }
  if( typeIndex == std::type_index( typeid( PyObject * ) ) )
  {
    return { NPY_OBJECT, sizeof( PyObject * ) };
  }
  return { NPY_NOTYPE, std::numeric_limits< std::size_t >::max() };
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * createNumpyArrayImpl( void * const data,
                                 std::type_index const type,
                                 bool const dataIsConst,
                                 int const ndim,
                                 long long const * const dims,
                                 long long const * const strides )
{
  if( !import_array_wrapper() )
  {
    return nullptr;
  }

  std::pair< int, std::size_t > const typeInfo = getNumPyType( type );

  PYTHON_ERROR_IF( typeInfo.first == NPY_NOTYPE, PyExc_TypeError,
                   "No NumPy type for " << system::demangle( type.name() ), nullptr );

  std::vector< npy_intp > byteStrides( ndim );
  std::vector< npy_intp > npyDims( ndim );

  for( int i = 0; i < ndim; ++i )
  {
    byteStrides[ i ] = integerConversion< npy_intp >( strides[ i ] * typeInfo.second );
    npyDims[ i ] = integerConversion< npy_intp >( dims[ i ] );
  }

  int const flags = dataIsConst ? 0 : NPY_ARRAY_WRITEABLE;
  return PyArray_NewFromDescr( &PyArray_Type,
                               PyArray_DescrFromType( typeInfo.first ),
                               ndim,
                               npyDims.data(),
                               byteStrides.data(),
                               data,
                               flags,
                               nullptr );
}


PyObject * createCupyArrayImpl( void * const data,
                                std::type_index const type,
                                bool const dataIsConst,
                                int const ndim,
                                long long const * const dims,
                                long long const * const strides )
{
  LVARRAY_UNUSED_VARIABLE( dataIsConst );
  if( !import_array_wrapper() )
  {
    return nullptr;
  }

  std::pair< int, std::size_t > const typeInfo = getNumPyType( type );
  PYTHON_ERROR_IF( typeInfo.first == NPY_NOTYPE, PyExc_TypeError,
                   "No NumPy type for " << system::demangle( type.name() ), nullptr );

  PyObjectRef<> typeObject{ PyArray_TypeObjectFromType( typeInfo.first ) };
  PYTHON_ERROR_IF( typeObject == nullptr, PyExc_TypeError, "Couldn't get type object", nullptr );

  PyObjectRef<> strideTuple{ PyTuple_New( integerConversion< Py_ssize_t >( ndim ) ) };
  PyObjectRef<> dimTuple{ PyTuple_New( integerConversion< Py_ssize_t >( ndim ) ) };
  if( strideTuple == nullptr || dimTuple == nullptr )
  {
    return nullptr;
  }

  Py_ssize_t totalSize = static_cast< Py_ssize_t >( typeInfo.second );
  for( Py_ssize_t i = 0; i < ndim; ++i )
  {
    totalSize = totalSize * dims[ i ];
    PyObjectRef<> strideEntry{ PyLong_FromSsize_t( integerConversion< Py_ssize_t >( strides[ i ] * typeInfo.second ) ) };
    PyObjectRef<> dimEntry{ PyLong_FromSsize_t( integerConversion< Py_ssize_t >( dims[ i ] ) ) };
    if( dimEntry == nullptr || strideEntry == nullptr )
    {
      return nullptr;
    }

    PyTuple_SET_ITEM( static_cast< PyObject * >( strideTuple ), i, strideEntry );
    PyTuple_SET_ITEM( static_cast< PyObject * >( dimTuple ), i, dimEntry );
  }

  PyObjectRef<> cupyHelper = PyImport_ImportModule( "cupy_helper" );
  if( cupyHelper == nullptr )
  {
    return nullptr;
  }

  PyObjectRef<> constructor = PyObject_GetAttrString( cupyHelper, "create_cupy_array" );
  if( constructor == nullptr )
  {
    return nullptr;
  }

  return PyObject_CallFunction( constructor,
                                "nnOOO",
                                (Py_ssize_t) ( data ),
                                totalSize,
                                static_cast< PyObject * >( typeObject ),
                                static_cast< PyObject * >( strideTuple ),
                                static_cast< PyObject * >( dimTuple ) );
}

} // namespace internal

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool import_array_wrapper()
{
  if( PyArray_API == nullptr )
  {
    import_array1( false );
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple< PyObjectRef< PyObject >, void const *, long long >
parseNumPyArray( PyObject * const obj, std::type_index const expectedType )
{
  using ReturnType = std::tuple< PyObjectRef< PyObject >, void const *, long long >;
  ReturnType const ErrorReturn{ nullptr, nullptr, 0 };

  if( !import_array_wrapper() )
  {
    return ErrorReturn;
  }

  PyObjectRef< PyArrayObject > array;
  if( PyArray_Converter( obj, reinterpret_cast< PyObject * * >( array.getAddress() ) ) == 0 )
  {
    return ErrorReturn;
  }

  PYTHON_ERROR_IF( PyArray_NDIM( array ) > 1, PyExc_RuntimeError,
                   "Expected a zero or one dimensional array.", ErrorReturn );

  PYTHON_ERROR_IF( !PyArray_ISCARRAY( array ), PyExc_RuntimeError,
                   "Data must be contiguous and well behaved.", ErrorReturn );

  int const srcNumPyType = PyArray_TYPE( array );
  int const expectedNumPyType = internal::getNumPyType( expectedType ).first;

  if( srcNumPyType == expectedNumPyType )
  {
    void const * vals = PyArray_DATA( array );
    long long const nVals = integerConversion< long long >( PyArray_SIZE( array ) );
    return ReturnType{ reinterpret_cast< PyObject * >( array.release() ), vals, nVals };
  }

  PYTHON_ERROR_IF( !PyArray_CanCastSafely( srcNumPyType, expectedNumPyType ), PyExc_TypeError,
                   "Cannot safely convert " << getNumPyTypeName( srcNumPyType ) <<
                   " to " << getNumPyTypeName( expectedNumPyType ), ErrorReturn );

  PyArrayObject * convertedArray = reinterpret_cast< PyArrayObject * >( PyArray_Cast( array, expectedNumPyType ) );
  void const * vals = PyArray_DATA( convertedArray );
  long long const nVals = integerConversion< long long >( PyArray_SIZE( convertedArray ) );
  return ReturnType{ reinterpret_cast< PyObject * >( convertedArray ), vals, nVals };
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::type_index getTypeIndexFromNumPy( int const numpyType )
{
  if( numpyType == NPY_BYTE )
  {
    return typeid( signed char );
  }
  if( numpyType == NPY_UBYTE )
  {
    return typeid( unsigned char );
  }
  if( numpyType == NPY_SHORT )
  {
    return typeid( short );
  }
  if( numpyType == NPY_USHORT )
  {
    return typeid( unsigned short );
  }
  if( numpyType == NPY_INT )
  {
    return typeid( int );
  }
  if( numpyType == NPY_UINT )
  {
    return typeid( unsigned int );
  }
  if( numpyType == NPY_LONG )
  {
    return typeid( long );
  }
  if( numpyType == NPY_ULONG )
  {
    return typeid( unsigned long );
  }
  if( numpyType == NPY_LONGLONG )
  {
    return typeid( long long );
  }
  if( numpyType == NPY_ULONGLONG )
  {
    return typeid( unsigned long long );
  }
  if( numpyType == NPY_FLOAT )
  {
    return typeid( float );
  }
  if( numpyType == NPY_DOUBLE )
  {
    return typeid( double );
  }
  if( numpyType == NPY_LONGDOUBLE )
  {
    return typeid( long double );
  }

  return typeid( nullptr );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string getNumPyTypeName( int const numpyType )
{
  if( numpyType == NPY_BYTE )
  {
    return "NPY_BYTE";
  }
  if( numpyType == NPY_UBYTE )
  {
    return "NPY_UBYTE";
  }
  if( numpyType == NPY_SHORT )
  {
    return "NPY_SHORT";
  }
  if( numpyType == NPY_USHORT )
  {
    return "NPY_USHORT";
  }
  if( numpyType == NPY_INT )
  {
    return "NPY_INT";
  }
  if( numpyType == NPY_UINT )
  {
    return "NPY_UINT";
  }
  if( numpyType == NPY_LONG )
  {
    return "NPY_LONG";
  }
  if( numpyType == NPY_ULONG )
  {
    return "NPY_ULONG";
  }
  if( numpyType == NPY_LONGLONG )
  {
    return "NPY_LONGLONG";
  }
  if( numpyType == NPY_ULONGLONG )
  {
    return "NPY_ULONGLONG";
  }
  if( numpyType == NPY_FLOAT )
  {
    return "NPY_FLOAT";
  }
  if( numpyType == NPY_DOUBLE )
  {
    return "NPY_DOUBLE";
  }
  if( numpyType == NPY_LONGDOUBLE )
  {
    return "NPY_LONGDOUBLE";
  }
  if( numpyType == NPY_OBJECT )
  {
    return "NPY_OBJECT";
  }

  return "Error: unrecognized type";
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * getNumPyTypeObject( std::type_index const typeIndex )
{
  PYTHON_ERROR_IF( !import_array_wrapper(), PyExc_RuntimeError, "Couldn't import array wrapper", nullptr );

  std::pair< int, std::size_t > const typeInfo = internal::getNumPyType( typeIndex );
  PYTHON_ERROR_IF( typeInfo.first == NPY_NOTYPE, PyExc_TypeError,
                   "No NumPy type for " << system::demangle( typeIndex.name() ), nullptr );

  PyObjectRef<> typeObject { PyArray_TypeObjectFromType( typeInfo.first ) };
  PYTHON_ERROR_IF( typeObject == nullptr, PyExc_TypeError, "Couldn't get type object", nullptr );

  return typeObject.release();
}

} // namespace python
} // namespace LvArray
