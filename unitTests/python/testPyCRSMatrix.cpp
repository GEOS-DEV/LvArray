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
#include <Python.h>

#include "python/PyCRSMatrix.hpp"
#include "python/python.hpp"
#include "MallocBuffer.hpp"

static LvArray::CRSMatrix< int, int, int, LvArray::MallocBuffer > matrixOfInts;
static LvArray::CRSMatrix< double, long, long, LvArray::MallocBuffer > matrixOfDoubles;

static PyObject * getMatrixOfInts( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );

  return LvArray::python::create( matrixOfInts );
}

static PyObject * getMatrixOfDoubles( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );

  return LvArray::python::create( matrixOfDoubles );
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef testPyCRSMatrixFuncs[] = {
  {"get_matrix_int", getMatrixOfInts, METH_NOARGS, ""},
  {"get_matrix_double", getMatrixOfDoubles, METH_NOARGS, ""},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPyCRSMatrixModule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "testPyCRSMatrix",
  .m_doc = "",
  .m_size = -1,
  .m_methods = testPyCRSMatrixFuncs,
};

END_ALLOW_DESIGNATED_INITIALIZERS

PyMODINIT_FUNC
PyInit_testPyCRSMatrix( void )
{
  LvArray::python::PyObjectRef<> module = PyModule_Create( &testPyCRSMatrixModule );

  if( !LvArray::python::addPyLvArrayModule( module ) )
  {
    return nullptr;
  }

  return module.release();
}
