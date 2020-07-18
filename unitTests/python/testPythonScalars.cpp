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

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL LvArray_ARRAY_API

#include "python/numpyHelpers.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

// global primitives
static signed char numpy_schar = -1;
static unsigned long numpy_ulong = 3;
static long double numpy_longdouble = 678;
static unsigned short numpy_ushort = 13;
static const short numpy_short_const = 17;
static bool numpy_bool = true; // should raise TypeError when converting
static char16_t numpy_char16 = 275; // should raise TypeError when converting

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_schar( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_schar );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_ulong( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_ulong );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_longdouble( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_longdouble );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_ushort( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_ushort );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_short_const( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_short_const );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_bool( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_bool );
}

/**
 * Fetch the global Scalar and return a numpy view of it.
 */
static PyObject *
get_char16( PyObject *self, PyObject *args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( numpy_char16 );
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef ScalarFuncs[] = {
  {"get_schar", get_schar, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_ulong", get_ulong, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_longdouble", get_longdouble, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_ushort", get_ushort, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_short_const", get_short_const, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_bool", get_bool, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {"get_char16", get_char16, METH_NOARGS,
   "Return the numpy representation of a global primitive."},
  {NULL, NULL, 0, NULL}          /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPythonScalarsmodule = {
  PyModuleDef_HEAD_INIT,
  "testPythonScalars",     /* name of module */
  /* module documentation, may be NULL */
  "Module for testing numpy views of scalars",
  -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
  ScalarFuncs,
  NULL,
  NULL,
  NULL,
  NULL,
};

PyMODINIT_FUNC
PyInit_testPythonScalars( void )
{
  import_array();
  return PyModule_Create( &testPythonScalarsmodule );
}
