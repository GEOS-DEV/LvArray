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
#include "PyArrayOfSets.hpp"
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
  PYTHON_ERROR_IF( self->arrayOfSets == nullptr, PyExc_RuntimeError, "The PyArrayOfSets is not initialized.", nullptr )

#define VERIFY_RESIZEABLE( self ) \
  PYTHON_ERROR_IF( self->arrayOfSets->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ), PyExc_RuntimeError, "The PyArrayOfSets is not resizeable.", nullptr )

struct PyArrayOfSets
{
  PyObject_HEAD

  static constexpr char const * docString =
    "Python interface to a LvArray::ArrayOfSets instance.\n\n"
    "Supports Python's sequence protocol, with the addition of\n"
    "deleting subsets with ``del arr[i]`` syntax.\n"
    "A set fetched with ``[]`` is returned as a read-only Numpy view.\n"
    "The built-in ``len()`` function will return the number of sets in the instance.\n"
    "Iterating over an instance will yield a Numpy view of each set.\n";

  internal::PyArrayOfSetsWrapperBase * arrayOfSets;
  PyObject * numpyDtype;
};

static void PyArrayOfSets_dealloc( PyArrayOfSets * const self )
{ delete self->arrayOfSets; Py_XDECREF( self->numpyDtype ); }

static PyObject * PyArrayOfSets_repr( PyObject * const obj )
{
  PyArrayOfSets * const self = convert< PyArrayOfSets >( obj, getPyArrayOfSetsType() );


  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  std::string const repr = self->arrayOfSets->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static Py_ssize_t PyArrayOfSets_len( PyArrayOfSets * const self )
{
  PYTHON_ERROR_IF( self->arrayOfSets == nullptr, PyExc_RuntimeError, "The PyArrayOfSets is not initialized.", -1 );
  return integerConversion< Py_ssize_t >( self->arrayOfSets->size() );
}

static PyObject * PyArrayOfSets_getitem( PyArrayOfSets * const self, Py_ssize_t pyindex )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  long long index = integerConversion< long long >( pyindex );
  if( PyErr_Occurred() != nullptr )
  {
    return nullptr;
  }
  if( index < 0 || index >= self->arrayOfSets->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return nullptr;
  }
  return self->arrayOfSets->operator[]( index );
}

static int PyArrayOfSets_delitem( PyArrayOfSets * const self, Py_ssize_t pyindex, PyObject * val )
{
  if( self->arrayOfSets->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
  {
    PyErr_SetString( PyExc_RuntimeError, "ArrayOfSets is not resizeable" );
    return -1;
  }
  if( val != nullptr )
  {
    PyErr_SetString( PyExc_TypeError, "ArrayOfSets object does not support item assignment" );
    return -1;
  }
  long long index = integerConversion< long long >( pyindex );
  if( PyErr_Occurred() != nullptr )
  {
    return -1;
  }
  if( index < 0 || index >= self->arrayOfSets->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return -1;
  }
  self->arrayOfSets->eraseSet( index );
  return 0;
}

static PyObject * PyArrayOfSets_sq_concat( PyArrayOfSets * self, PyObject * args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static PyObject * PyArrayOfSets_sq_repeat( PyArrayOfSets * self, Py_ssize_t args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static constexpr char const * PyArrayOfSets_insertIntoDocString =
  "insert_into(self, set, values)\n"
  "--\n\n"
  "Insert values into a set"
  "Returns\n"
  "_______\n"
  "integer\n"
  "    The number of values inserted into the set.";
static PyObject * PyArrayOfSets_insertIntoSet( PyArrayOfSets * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long setIndex;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LO", &setIndex, &arr ) )
  {
    return nullptr;
  }

  if( setIndex < 0 || setIndex >= self->arrayOfSets->size() )
  {
    PyErr_SetString( PyExc_IndexError, "index out of bounds" );
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( arr, self->arrayOfSets->valueType() );

  if( std::get< 0 >( ret ) == nullptr )
  {
    return nullptr;
  }

  return PyLong_FromLongLong( self->arrayOfSets->insertIntoSet( setIndex,
                                                                std::get< 1 >( ret ),
                                                                std::get< 2 >( ret ) ) );
}

static constexpr char const * PyArrayOfSets_insertSetDocString =
  "insert(self, position, capacity=0)\n"
  "--\n\n"
  "Insert a new set with a starting capacity";
static PyObject * PyArrayOfSets_insertSet( PyArrayOfSets * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long index;
  long long capacity = 0;
  if( !PyArg_ParseTuple( args, "L|L", &index, &capacity ) )
  {
    return nullptr;
  }
  if( index < 0 || index > self->arrayOfSets->size() )
  {
    PyErr_SetString( PyExc_IndexError, "Index out of bounds" );
    return nullptr;
  }
  if( capacity < 0 )
  {
    PyErr_SetString( PyExc_ValueError, "capacity must be positive" );
    return nullptr;
  }
  self->arrayOfSets->insertSet( index, capacity );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfSets_removeFromSetDocString =
  "erase_from(self, set_index, values)\n"
  "--\n\n"
  "Erase values from a set."
  "Returns\n"
  "_______\n"
  "integer\n"
  "    The number of values removed from the set.";
static PyObject * PyArrayOfSets_removeFromSet( PyArrayOfSets * self, PyObject * args )
{
  VERIFY_RESIZEABLE( self );
  long long index;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LO", &index, &arr ) )
  {
    return nullptr;
  }

  if( index < 0 || index >= self->arrayOfSets->size() )
  {
    PyErr_SetString( PyExc_IndexError, "index out of bounds" );
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( arr, self->arrayOfSets->valueType() );

  return PyLong_FromLongLong( self->arrayOfSets->removeFromSet( index,
                                                                std::get< 1 >( ret ),
                                                                std::get< 2 >( ret ) ) );
}

static constexpr char const * PyArrayOfSets_getAccessLevelDocString =
  "get_access_level(self)\n"
  "--\n\n"
  "Return the read/write/resize permissions for the instance.";
static PyObject * PyArrayOfSets_getAccessLevel( PyArrayOfSets * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );

  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLong( self->arrayOfSets->getAccessLevel() );
}

static constexpr char const * PyArrayOfSets_setAccessLevelDocString =
  "set_access_level(self, level)\n"
  "--\n\n"
  "Set read/write/resize permissions for the instance.\n"
  "The ``MODIFIABLE`` permission has no effect.";
static PyObject * PyArrayOfSets_setAccessLevel( PyArrayOfSets * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  int newAccessLevel;
  if( !PyArg_ParseTuple( args, "i", &newAccessLevel ) )
  {
    return nullptr;
  }
  self->arrayOfSets->setAccessLevel( newAccessLevel );
  Py_RETURN_NONE;
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMemberDef PyArrayOfSets_members[] = {
  {"dtype", T_OBJECT_EX, offsetof( PyArrayOfSets, numpyDtype ), READONLY,
   "Numpy dtype of the object"},
  {nullptr, 0, 0, 0, nullptr}    /* Sentinel */
};

static PyMethodDef PyArrayOfSets_methods[] = {
  { "insert", (PyCFunction) PyArrayOfSets_insertSet, METH_VARARGS, PyArrayOfSets_insertSetDocString },
  { "insert_into", (PyCFunction) PyArrayOfSets_insertIntoSet, METH_VARARGS, PyArrayOfSets_insertIntoDocString },
  { "erase_from", (PyCFunction) PyArrayOfSets_removeFromSet, METH_VARARGS, PyArrayOfSets_removeFromSetDocString },
  { "get_access_level", (PyCFunction) PyArrayOfSets_getAccessLevel, METH_NOARGS, PyArrayOfSets_getAccessLevelDocString },
  { "set_access_level", (PyCFunction) PyArrayOfSets_setAccessLevel, METH_VARARGS, PyArrayOfSets_setAccessLevelDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PySequenceMethods PyArrayOfSetsSequenceMethods = {
  .sq_length = (lenfunc) PyArrayOfSets_len,
  .sq_concat = (binaryfunc) PyArrayOfSets_sq_concat,
  .sq_repeat = (ssizeargfunc) PyArrayOfSets_sq_repeat,
  .sq_item = (ssizeargfunc) PyArrayOfSets_getitem,
  .sq_ass_item = (ssizeobjargproc) PyArrayOfSets_delitem,
};

static PyTypeObject PyArrayOfSetsType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
    .tp_name = "pylvarray.ArrayOfSets",
  .tp_basicsize = sizeof( PyArrayOfSets ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyArrayOfSets_dealloc,
  .tp_repr = PyArrayOfSets_repr,
  .tp_as_sequence = &PyArrayOfSetsSequenceMethods,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyArrayOfSets::docString,
  .tp_methods = PyArrayOfSets_methods,
  .tp_members = PyArrayOfSets_members,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyArrayOfSetsType()
{ return &PyArrayOfSetsType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyArrayOfSetsWrapperBase > && arrayOfSets )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyArrayOfSetsType() ), "" );
  PyArrayOfSets * const retArrayOfSets = reinterpret_cast< PyArrayOfSets * >( ret );
  if( retArrayOfSets == nullptr )
  {
    return nullptr;
  }

  PyObject * typeObject = getNumPyTypeObject( arrayOfSets->valueType() );
  if( typeObject == nullptr )
  {
    return nullptr;
  }

  retArrayOfSets->numpyDtype = typeObject;
  retArrayOfSets->arrayOfSets = arrayOfSets.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray

/// @endcond DO_NOT_DOCUMENT
