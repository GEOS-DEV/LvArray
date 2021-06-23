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
#include "PyCRSMatrix.hpp"
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
  PYTHON_ERROR_IF( self->matrix == nullptr, PyExc_RuntimeError, "The PyCRSMatrix is not initialized.", nullptr )

#define VERIFY_RESIZEABLE( self ) \
  PYTHON_ERROR_IF( self->matrix->getAccessLevel() < static_cast< int >( LvArray::python::PyModify::RESIZEABLE ), PyExc_RuntimeError, "The PyCRSMatrix is not resizeable.", nullptr )

struct PyCRSMatrix
{
  PyObject_HEAD

  static constexpr char const * docString =
    "Represents a LvArray::CRSMatrix instance.\n\n";

  internal::PyCRSMatrixWrapperBase * matrix;

  PyObject * numpyDtype;
};

static void PyCRSMatrix_dealloc( PyCRSMatrix * const self )
{ delete self->matrix; Py_XDECREF( self->numpyDtype ); }

static PyObject * PyCRSMatrix_repr( PyObject * const obj )
{
  PyCRSMatrix * const self = convert< PyCRSMatrix >( obj, getPyCRSMatrixType() );

  if( self == nullptr )
  {
    return nullptr;
  }

  VERIFY_INITIALIZED( self );

  std::string const repr = self->matrix->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static constexpr char const * PyCRSMatrix_numRowsDocString =
  "num_rows(self)\n"
  "--\n\n"
  "Return the number of rows in the matrix.";
static PyObject * PyCRSMatrix_numRows( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLongLong( self->matrix->numRows() );
}

static constexpr char const * PyCRSMatrix_numColumnsDocString =
  "num_columns(self)\n"
  "--\n\n"
  "Return the number of columns in the matrix.";
static PyObject * PyCRSMatrix_numColumns( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLongLong( self->matrix->numColumns() );
}

static constexpr char const * PyCRSMatrix_getColumnsDocString =
  "get_columns(self, row)\n"
  "--\n\n";
static PyObject * PyCRSMatrix_getColumns( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  long row;
  if( !PyArg_ParseTuple( args, "l", &row ) )
  {
    return nullptr;
  }

  long long const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row  <<" is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return self->matrix->getColumns( row );
}

static constexpr char const * PyCRSMatrix_getEntriesDocString =
  "get_entries(self, row)\n"
  "--\n\n";
static PyObject * PyCRSMatrix_getEntries( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  long row;
  if( !PyArg_ParseTuple( args, "l", &row ) )
  {
    return nullptr;
  }

  long long const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return self->matrix->getEntries( row );
}

static constexpr char const * PyCRSMatrix_toSciPyDocString =
  "to_scipy(self)\n"
  "--\n\n"
  "Return a ``scipy.sparse.csr_matrix`` view of the instance.";
static PyObject * PyCRSMatrix_toSciPy( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  PYTHON_ERROR_IF( !self->matrix->isCompressed(), PyExc_RuntimeError,
                   "Uncompressed matrices can not be exported to SciPy", nullptr );

  PYTHON_ERROR_IF( self->matrix->numRows() == 0, PyExc_RuntimeError,
                   "Empty matrices cannot be exported to SciPy", nullptr );

  std::array< PyObject *, 3 > const triple = self->matrix->getEntriesColumnsAndOffsets();

  PYTHON_ERROR_IF( triple[ 0 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the entries NumPy array", nullptr );

  PYTHON_ERROR_IF( triple[ 1 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the columns NumPy array", nullptr );

  PYTHON_ERROR_IF( triple[ 2 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the offsets NumPy array", nullptr );

  PyObjectRef<> sciPySparse = PyImport_ImportModule( "scipy.sparse" );
  PyObjectRef<> constructor = PyObject_GetAttrString( sciPySparse, "csr_matrix" );

  return PyObject_CallFunction( constructor,
                                "(OOO)(ll)",
                                triple[ 0 ],
                                triple[ 1 ],
                                triple[ 2 ],
                                self->matrix->numRows(),
                                self->matrix->numColumns() );
}

static constexpr char const * PyCRSMatrix_resizeDocString =
  "resize(self, num_rows, num_cols, initial_row_capacity=0)\n"
  "--\n\n"
  "Set the dimensions of the matrix.";
static PyObject * PyCRSMatrix_resize( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  long numRows, numCols, initialRowCapacity=0;
  if( !PyArg_ParseTuple( args, "ll|l", &numRows, &numCols, &initialRowCapacity ) )
  {
    return nullptr;
  }

  PYTHON_ERROR_IF( numRows < 0 || numCols < 0 || initialRowCapacity < 0, PyExc_RuntimeError,
                   "Arguments must be positive.", nullptr );

  self->matrix->resize( numRows, numCols, initialRowCapacity );

  Py_RETURN_NONE;
}

static constexpr char const * PyCRSMatrix_compressDocString =
  "compress(self)\n"
  "--\n\n";
static PyObject * PyCRSMatrix_compress( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  self->matrix->compress();

  Py_RETURN_NONE;
}

static constexpr char const * PyCRSMatrix_insertNonZerosDocString =
  "insert_nonzeros(self, row, columns, entries)\n"
  "--\n\n"
  "Insert nonzero entries into the given row.";
static PyObject * PyCRSMatrix_insertNonZeros( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  long row;
  PyObject * cols, * entries;
  if( !PyArg_ParseTuple( args, "lOO", &row, &cols, &entries ) )
  {
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  std::tuple< PyObjectRef< PyObject >, void const *, long long > entriesInfo =
    parseNumPyArray( entries, self->matrix->entryType() );

  if( std::get< 0 >( colsInfo ) == nullptr ||  std::get< 0 >( entriesInfo ) == nullptr )
  {
    return nullptr;
  }

  PYTHON_ERROR_IF( std::get< 2 >( colsInfo ) != std::get< 2 >( entriesInfo ), PyExc_RuntimeError,
                   "Number of columns and entries must match!", nullptr );

  long long const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return PyLong_FromLongLong( self->matrix->insertNonZeros( row,
                                                            std::get< 1 >( colsInfo ),
                                                            std::get< 1 >( entriesInfo ),
                                                            std::get< 2 >( entriesInfo ) ) );
}

static constexpr char const * PyCRSMatrix_removeNonZerosDocString =
  "remove_nonzeros(self, row, columns)\n"
  "--\n\n"
  "Remove nonzero entries from the matrix.\n\n"
  "`columns` should be an iterable identifying the columns of"
  "the given row to remove nonzero entries from.";
static PyObject * PyCRSMatrix_removeNonZeros( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  long row;
  PyObject * cols;
  if( !PyArg_ParseTuple( args, "lO", &row, &cols ) )
  {
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  if( std::get< 0 >( colsInfo ) == nullptr )
  {
    return nullptr;
  }

  long long const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return PyLong_FromLongLong( self->matrix->removeNonZeros( row,
                                                            std::get< 1 >( colsInfo ),
                                                            std::get< 2 >( colsInfo ) ) );
}

static constexpr char const * PyCRSMatrix_addToRowDocString =
  "add_to_row(self, row, columns, values)\n"
  "--\n\n"
  "Add an iterable of values to a row.\n\n"
  "The entries should already exist in the matrix.\n"
  "`columns` and `values` should be iterables of equal length.\n";
static PyObject * PyCRSMatrix_addToRow( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_RESIZEABLE( self );

  long row;
  PyObject * cols, * vals;
  if( !PyArg_ParseTuple( args, "lOO", &row, &cols, &vals ) )
  {
    return nullptr;
  }

  std::tuple< PyObjectRef< PyObject >, void const *, long long > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  std::tuple< PyObjectRef< PyObject >, void const *, long long > valsInfo =
    parseNumPyArray( vals, self->matrix->entryType() );

  if( std::get< 0 >( colsInfo ) == nullptr ||  std::get< 0 >( valsInfo ) == nullptr )
  {
    return nullptr;
  }

  long long const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  self->matrix->addToRow( row,
                          std::get< 1 >( colsInfo ),
                          std::get< 1 >( valsInfo ),
                          std::get< 2 >( valsInfo ) );

  Py_RETURN_NONE;
}

static constexpr char const * PyCRSMatrix_getAccessLevelDocString =
  "get_access_level(self)\n"
  "--\n\n"
  "Return the read/write/resize permissions for the instance.";
static PyObject * PyCRSMatrix_getAccessLevel( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLong( self->matrix->getAccessLevel() );
}

static constexpr char const * PyCRSMatrix_setAccessLevelDocString =
  "set_access_level(self, level, space=pylvarray.CPU)\n"
  "--\n\n"
  "Set read/write/resize permissions for the instance. "
  "If ``space`` is provided, move the instance to that space.";
static PyObject * PyCRSMatrix_setAccessLevel( PyCRSMatrix * const self, PyObject * const args )
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
  self->matrix->setAccessLevel( newAccessLevel, newSpace );
  Py_RETURN_NONE;
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMemberDef PyCRSMatrix_members[] = {
  {"dtype", T_OBJECT_EX, offsetof( PyCRSMatrix, numpyDtype ), READONLY,
   "Numpy dtype of the object"},
  {nullptr, 0, 0, 0, nullptr}    /* Sentinel */
};

static PyMethodDef PyCRSMatrix_methods[] = {
  { "num_rows", (PyCFunction) PyCRSMatrix_numRows, METH_NOARGS, PyCRSMatrix_numRowsDocString },
  { "num_columns", (PyCFunction) PyCRSMatrix_numColumns, METH_NOARGS, PyCRSMatrix_numColumnsDocString },
  { "get_columns", (PyCFunction) PyCRSMatrix_getColumns, METH_VARARGS, PyCRSMatrix_getColumnsDocString },
  { "get_entries", (PyCFunction) PyCRSMatrix_getEntries, METH_VARARGS, PyCRSMatrix_getEntriesDocString },
  { "to_scipy", (PyCFunction) PyCRSMatrix_toSciPy, METH_NOARGS, PyCRSMatrix_toSciPyDocString },
  { "resize", (PyCFunction) PyCRSMatrix_resize, METH_VARARGS, PyCRSMatrix_resizeDocString },
  { "compress", (PyCFunction) PyCRSMatrix_compress, METH_NOARGS, PyCRSMatrix_compressDocString },
  { "insert_nonzeros", (PyCFunction) PyCRSMatrix_insertNonZeros, METH_VARARGS, PyCRSMatrix_insertNonZerosDocString },
  { "remove_nonzeros", (PyCFunction) PyCRSMatrix_removeNonZeros, METH_VARARGS, PyCRSMatrix_removeNonZerosDocString },
  { "add_to_row", (PyCFunction) PyCRSMatrix_addToRow, METH_VARARGS, PyCRSMatrix_addToRowDocString },
  { "get_access_level", (PyCFunction) PyCRSMatrix_getAccessLevel, METH_NOARGS, PyCRSMatrix_getAccessLevelDocString },
  { "set_access_level", (PyCFunction) PyCRSMatrix_setAccessLevel, METH_VARARGS, PyCRSMatrix_setAccessLevelDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyTypeObject PyCRSMatrixType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
    .tp_name = "pylvarray.CRSMatrix",
  .tp_basicsize = sizeof( PyCRSMatrix ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyCRSMatrix_dealloc,
  .tp_repr = PyCRSMatrix_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyCRSMatrix::docString,
  .tp_methods = PyCRSMatrix_methods,
  .tp_members = PyCRSMatrix_members,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyCRSMatrixType()
{ return &PyCRSMatrixType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyCRSMatrixWrapperBase > && matrix )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyCRSMatrixType() ), "" );
  PyCRSMatrix * const retMatrix = reinterpret_cast< PyCRSMatrix * >( ret );
  if( retMatrix == nullptr )
  {
    return nullptr;
  }

  PyObject * typeObject = getNumPyTypeObject( matrix->entryType() );
  if( typeObject == nullptr )
  {
    return nullptr;
  }

  retMatrix->numpyDtype = typeObject;
  retMatrix->matrix = matrix.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray

/// @endcond DO_NOT_DOCUMENT
