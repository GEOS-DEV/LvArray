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
// Python include for converting struct attributes to Python objects
#include "structmember.h"

// Source includes
#include "PyArrayOfArrays.hpp"
#include "pythonHelpers.hpp"
#include "../limits.hpp"

// System includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

/// @cond DO_NOT_DOCUMENT

namespace LvArray
{
namespace python
{

#define VERIFY_NON_NULL_SELF( self ) \
  PYTHON_ERROR_IF( self == nullptr, PyExc_RuntimeError, "Passed a nullptr as self.", nullptr )

#define VERIFY_INITIALIZED( self ) \
  PYTHON_ERROR_IF( self->arrayOfArrays == nullptr, PyExc_RuntimeError, "The PyArrayOfArrays is not initialized.", nullptr )

#define VERIFY_RESIZEABLE( self ) \
  PYTHON_ERROR_IF( self->arrayOfArrays->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ), PyExc_RuntimeError, "The PyArrayOfArrays is not resizeable.", nullptr )

struct PyArrayOfArrays
{
  PyObject_HEAD

  static constexpr char const * docString =
    "Python interface to a LvArray::ArrayOfArrays instance.\n\n"
    "Supports Python's sequence protocol, with the addition of\n"
    "deleting subsets with ``del arr[i]`` syntax.\n"
    "An array fetched with ``[]`` is returned as a Numpy view.\n"
    "The built-in ``len()`` function will return the number of arrays in the instance.\n"
    "Iterating over an instance will yield a Numpy view of each array.";

  internal::PyArrayOfArraysWrapperBase * arrayOfArrays;
  PyObject * numpyDtype;
};

static void PyArrayOfArrays_dealloc( PyArrayOfArrays * const self )
{ delete self->arrayOfArrays; Py_XDECREF( self->numpyDtype ); }

static PyObject * PyArrayOfArrays_repr( PyObject * const obj )
{
  PyArrayOfArrays * const self = convert< PyArrayOfArrays >( obj, getPyArrayOfArraysType() );


  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  std::string const repr = self->arrayOfArrays->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static Py_ssize_t PyArrayOfArrays_len( PyArrayOfArrays * const self )
{
  PYTHON_ERROR_IF( self->arrayOfArrays == nullptr, PyExc_RuntimeError, "The PyArrayOfArrays is not initialized.", -1 );
  return integerConversion< Py_ssize_t >( self->arrayOfArrays->size() );
}

static PyObject * PyArrayOfArrays_getitem( PyArrayOfArrays * const self, Py_ssize_t pyindex )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  long long index = integerConversion< long long >( pyindex );
  if( PyErr_Occurred() != nullptr )
  {
    return nullptr;
  }
  if( index < 0 || index >= self->arrayOfArrays->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return nullptr;
  }
  return self->arrayOfArrays->operator[]( index );
}

static int PyArrayOfArrays_delitem( PyArrayOfArrays * const self, Py_ssize_t pyindex, PyObject * val )
{
  if( self->arrayOfArrays->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
  {
    PyErr_SetString( PyExc_RuntimeError, "ArrayOfArrays is not resizeable" );
    return -1;
  }
  if( val != nullptr )
  {
    PyErr_SetString( PyExc_TypeError, "ArrayOfArrays object does not support item assignment" );
    return -1;
  }
  long long index = integerConversion< long long >( pyindex );
  if( PyErr_Occurred() != nullptr )
  {
    return -1;
  }
  if( index < 0 || index >= self->arrayOfArrays->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return -1;
  }
  self->arrayOfArrays->eraseArray( index );
  return 0;
}

static PyObject * PyArrayOfArrays_sq_concat( PyArrayOfArrays * self, PyObject * args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static PyObject * PyArrayOfArrays_sq_repeat( PyArrayOfArrays * self, Py_ssize_t args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static constexpr char const * PyArrayOfArrays_insertIntoDocString =
  "insert_into(self, array, index, values)\n"
  "--\n\n"
  "Insert ``values`` into the subarray ``array`` at position ``index``.\n"
  "``values`` will be converted to a 1D numpy array of the same dtype\n"
  "as the underlying instance, raising an exception if the conversion cannot be made safely.";
static PyObject * PyArrayOfArrays_insertIntoArray( PyArrayOfArrays * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long array, index;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LLO", &array, &index, &arr ) )
  {
    return nullptr;
  }
  if( array < 0 || array >= self->arrayOfArrays->size() || index < 0
      || index > self->arrayOfArrays->sizeOfArray( array ))
  {
    PyErr_SetString( PyExc_IndexError, "index out of bounds" );
    return nullptr;
  }
  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( arr, self->arrayOfArrays->valueType() );
  if( std::get< 0 >( ret ) == nullptr )
  {
    return nullptr;
  }
  self->arrayOfArrays->insertIntoArray( array, index, std::get< 1 >( ret ), std::get< 2 >( ret ) );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfArrays_insertDocString =
  "insert(self, position, values)\n"
  "--\n\n"
  "Insert a new array consisting of ``values`` at a specific position.\n"
  "``values`` will be converted to a 1D numpy array of the same dtype\n"
  "as the underlying instance, raising an exception if the conversion cannot be made safely.";
static PyObject * PyArrayOfArrays_insert( PyArrayOfArrays * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long index;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LO", &index, &arr ) )
  {
    return nullptr;
  }
  if( index < 0 || index > self->arrayOfArrays->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return nullptr;
  }
  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( arr, self->arrayOfArrays->valueType() );
  if( std::get< 0 >( ret ) == nullptr )
  {
    return nullptr;
  }
  self->arrayOfArrays->insertArray( index, std::get< 1 >( ret ), std::get< 2 >( ret ) );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfArrays_eraseFromDocString =
  "erase_from(self, array, index)\n"
  "--\n\n"
  "Remove the value at ``index`` in the subarray ``array``.";
static PyObject * PyArrayOfArrays_eraseFrom( PyArrayOfArrays * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long index, begin;
  if( !PyArg_ParseTuple( args, "LL", &index, &begin ) )
  {
    return nullptr;
  }
  if( index < 0 || index >= self->arrayOfArrays->size() || begin < 0
      || begin >= self->arrayOfArrays->sizeOfArray( index ) )
  {
    PyErr_SetString( PyExc_IndexError, "index out of bounds" );
    return nullptr;
  }
  self->arrayOfArrays->eraseFromArray( index, begin );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfArrays_getAccessLevelDocString =
  "get_access_level(self)\n"
  "--\n\n"
  "Return the read/write/resize permissions for the instance.";
static PyObject * PyArrayOfArrays_getAccessLevel( PyArrayOfArrays * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );

  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLong( self->arrayOfArrays->getAccessLevel() );
}

static constexpr char const * PyArrayOfArrays_setAccessLevelDocString =
  "set_access_level(self, level)\n"
  "--\n\n"
  "Set read/write/resize permissions for the instance.";
static PyObject * PyArrayOfArrays_setAccessLevel( PyArrayOfArrays * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  int newAccessLevel;
  if( !PyArg_ParseTuple( args, "i", &newAccessLevel ) )
  {
    return nullptr;
  }
  self->arrayOfArrays->setAccessLevel( newAccessLevel );
  Py_RETURN_NONE;
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMemberDef PyArrayOfArrays_members[] = {
  {"dtype", T_OBJECT_EX, offsetof( PyArrayOfArrays, numpyDtype ), READONLY,
   "Numpy dtype of the object"},
  {nullptr, 0, 0, 0, nullptr}    /* Sentinel */
};

static PyMethodDef PyArrayOfArrays_methods[] = {
  { "insert", (PyCFunction) PyArrayOfArrays_insert, METH_VARARGS, PyArrayOfArrays_insertDocString },
  { "insert_into", (PyCFunction) PyArrayOfArrays_insertIntoArray, METH_VARARGS, PyArrayOfArrays_insertIntoDocString },
  { "erase_from", (PyCFunction) PyArrayOfArrays_eraseFrom, METH_VARARGS, PyArrayOfArrays_eraseFromDocString },
  { "get_access_level", (PyCFunction) PyArrayOfArrays_getAccessLevel, METH_NOARGS, PyArrayOfArrays_getAccessLevelDocString },
  { "set_access_level", (PyCFunction) PyArrayOfArrays_setAccessLevel, METH_VARARGS, PyArrayOfArrays_setAccessLevelDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PySequenceMethods PyArrayOfArraysSequenceMethods = {
  .sq_length = (lenfunc) PyArrayOfArrays_len,
  .sq_concat = (binaryfunc) PyArrayOfArrays_sq_concat,
  .sq_repeat = (ssizeargfunc) PyArrayOfArrays_sq_repeat,
  .sq_item = (ssizeargfunc) PyArrayOfArrays_getitem,
  .sq_ass_item = (ssizeobjargproc) PyArrayOfArrays_delitem,
};

static PyTypeObject PyArrayOfArraysType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
    .tp_name = "pylvarray.ArrayOfArrays",
  .tp_basicsize = sizeof( PyArrayOfArrays ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyArrayOfArrays_dealloc,
  .tp_repr = PyArrayOfArrays_repr,
  .tp_as_sequence = &PyArrayOfArraysSequenceMethods,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyArrayOfArrays::docString,
  .tp_methods = PyArrayOfArrays_methods,
  .tp_members = PyArrayOfArrays_members,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyArrayOfArraysType()
{ return &PyArrayOfArraysType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyArrayOfArraysWrapperBase > && arrayOfArrays )
{
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyArrayOfArraysType() ), "" );
  PyArrayOfArrays * const retArrayOfArrays = reinterpret_cast< PyArrayOfArrays * >( ret );
  if( retArrayOfArrays == nullptr )
  {
    return nullptr;
  }

  PyObject * typeObject = getNumPyTypeObject( arrayOfArrays->valueType() );
  if( typeObject == nullptr )
  {
    return nullptr;
  }

  retArrayOfArrays->numpyDtype = typeObject;
  retArrayOfArrays->arrayOfArrays = arrayOfArrays.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray

/// @endcond DO_NOT_DOCUMENT
