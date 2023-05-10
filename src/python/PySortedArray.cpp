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
#include "PySortedArray.hpp"
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
  PYTHON_ERROR_IF( self->sortedArray == nullptr, PyExc_RuntimeError, "The PySortedArray is not initialized.", nullptr )

#define VERIFY_RESIZEABLE( self ) \
  PYTHON_ERROR_IF( self->sortedArray->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ), PyExc_RuntimeError, "The PySortedArray is not resizeable.", nullptr )

struct PySortedArray
{
  PyObject_HEAD

  static constexpr char const * docString =
    "Represents a LvArray::SortedArray instance.\n\n";

  internal::PySortedArrayWrapperBase * sortedArray;
  PyObject * numpyDtype;
};


static void PySortedArray_dealloc( PySortedArray * const self )
{ delete self->sortedArray; Py_XDECREF( self->numpyDtype ); }


static PyObject * PySortedArray_repr( PyObject * const obj )
{
  PySortedArray * const self = convert< PySortedArray >( obj, getPySortedArrayType() );

  if( self == nullptr )
  {
    return nullptr;
  }

  VERIFY_INITIALIZED( self );

  std::string const repr = self->sortedArray->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static constexpr char const * PySortedArray_insertDocString =
  "insert(self, values)\n"
  "--\n\n"
  "Insert one or more values into the array.\n"
  "The object passed in will be converted to a 1D Numpy array of the same dtype\n"
  "as the underlying instance, raising an exception if the conversion cannot be made safely.";
static PyObject * PySortedArray_insert( PySortedArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  PyObject * obj;
  if( !PyArg_ParseTuple( args, "O", &obj ) )
  {
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( obj, self->sortedArray->valueType() );

  if( std::get< 0 >( ret ) == nullptr )
  {
    return nullptr;
  }

  return PyLong_FromLongLong( self->sortedArray->insert( std::get< 1 >( ret ), std::get< 2 >( ret ) ) );
}

static constexpr char const * PySortedArray_removeDocString =
  "remove(self, values)\n"
  "--\n\n"
  "Remove one or more values from the array.\n"
  "The object passed in will be converted to a 1D numpy array of the same dtype\n"
  "as the underlying instance, raising an exception if the conversion cannot be made safely.";
static PyObject * PySortedArray_remove( PySortedArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  PyObject * obj;
  if( !PyArg_ParseTuple( args, "O", &obj ) )
  {
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > ret =
    parseNumPyArray( obj, self->sortedArray->valueType() );

  if( std::get< 0 >( ret ) == nullptr )
  {
    return nullptr;
  }

  return PyLong_FromLongLong( self->sortedArray->remove( std::get< 1 >( ret ), std::get< 2 >( ret ) ) );
}


static constexpr char const * PySortedArray_toNumPyDocString =
  "to_numpy(self)\n"
  "--\n\n"
  "Return a read-only Numpy view of the instance.";
static PyObject * PySortedArray_toNumPy( PySortedArray * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );

  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return self->sortedArray->toNumPy();
}

static constexpr char const * PySortedArray_getAccessLevelDocString =
  "get_access_level(self)\n"
  "--\n\n"
  "Return the read/write/resize permissions for the instance.";
static PyObject * PySortedArray_getAccessLevel( PySortedArray * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLong( self->sortedArray->getAccessLevel() );
}

static constexpr char const * PySortedArray_setAccessLevelDocString =
  "set_access_level(self, level, space=pylvarray.CPU)\n"
  "--\n\n"
  "Set read/write/resize permissions for the instance.\n"
  "The ``MODIFIABLE`` permission has no effect, because the data\n"
  "must be maintained in sorted order.\n"
  "If ``space`` is provided, move the instance to that space.";
static PyObject * PySortedArray_setAccessLevel( PySortedArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  int newAccessLevel;
  int newSpace = static_cast< int >( LvArray::MemorySpace::host );
  if( !PyArg_ParseTuple( args, "i|i", &newAccessLevel, &newSpace ) )
  {
    return nullptr;
  }
  if( newSpace != static_cast< int >( LvArray::MemorySpace::host ) )
  {
    PyErr_SetString( PyExc_ValueError, "Invalid space" );
    return nullptr;
  }
  self->sortedArray->setAccessLevel( newAccessLevel, newSpace );
  Py_RETURN_NONE;
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMemberDef PySortedArray_members[] = {
  {"dtype", T_OBJECT_EX, offsetof( PySortedArray, numpyDtype ), READONLY,
   "Numpy dtype of the object"},
  {nullptr, 0, 0, 0, nullptr}    /* Sentinel */
};

static PyMethodDef PySortedArray_methods[] = {
  { "insert", (PyCFunction) PySortedArray_insert, METH_VARARGS, PySortedArray_insertDocString },
  { "remove", (PyCFunction) PySortedArray_remove, METH_VARARGS, PySortedArray_removeDocString },
  { "to_numpy", (PyCFunction) PySortedArray_toNumPy, METH_VARARGS, PySortedArray_toNumPyDocString },
  { "get_access_level", (PyCFunction) PySortedArray_getAccessLevel, METH_NOARGS, PySortedArray_getAccessLevelDocString },
  { "set_access_level", (PyCFunction) PySortedArray_setAccessLevel, METH_VARARGS, PySortedArray_setAccessLevelDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyTypeObject PySortedArrayType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
    .tp_name = "pylvarray.SortedArray",
  .tp_basicsize = sizeof( PySortedArray ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PySortedArray_dealloc,
  .tp_repr = PySortedArray_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PySortedArray::docString,
  .tp_methods = PySortedArray_methods,
  .tp_members = PySortedArray_members,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPySortedArrayType()
{ return &PySortedArrayType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PySortedArrayWrapperBase > && sortedArray )
{
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPySortedArrayType() ), "" );
  PySortedArray * const retSortedArray = reinterpret_cast< PySortedArray * >( ret );
  if( retSortedArray == nullptr )
  {
    return nullptr;
  }


  PyObject * typeObject = getNumPyTypeObject( sortedArray->valueType() );
  if( typeObject == nullptr )
  {
    return nullptr;
  }

  retSortedArray->numpyDtype = typeObject;
  retSortedArray->sortedArray = sortedArray.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray

/// @endcond DO_NOT_DOCUMENT
