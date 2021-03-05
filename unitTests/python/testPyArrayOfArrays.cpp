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

#include "python/PyArrayOfArrays.hpp"
#include "python/python.hpp"
#include "MallocBuffer.hpp"

namespace
{

static LvArray::ArrayOfArrays< long, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays{ 5, 6 };

} // namespace

static PyObject * getArrayOfArrays( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( arrayOfArrays );
}


BEGIN_ALLOW_DESIGNATED_INITIALIZERS

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef testPyArrayOfArraysFuncs[] = {
  {"get_array_of_arrays", getArrayOfArrays, METH_NOARGS, ""},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPyArrayOfArraysModule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "testPyArrayOfArrays",
  .m_doc = "",
  .m_size = -1,
  .m_methods = testPyArrayOfArraysFuncs,
};

END_ALLOW_DESIGNATED_INITIALIZERS

PyMODINIT_FUNC
PyInit_testPyArrayOfArrays( void )
{
  LvArray::python::PyObjectRef<> module = PyModule_Create( &testPyArrayOfArraysModule );

  if( !LvArray::python::addPyLvArrayModule( module ) )
  {
    return nullptr;
  }

  return module.release();
}
