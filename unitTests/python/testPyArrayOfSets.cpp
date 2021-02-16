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

#include "python/PyArrayOfSets.hpp"
#include "python/python.hpp"
#include "MallocBuffer.hpp"

namespace
{

static LvArray::ArrayOfSets< long long, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfSets{ 5, 6 };

} // namespace

static PyObject * getArrayOfSets( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  return LvArray::python::create( arrayOfSets );
}


BEGIN_ALLOW_DESIGNATED_INITIALIZERS

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef testPyArrayOfSetsFuncs[] = {
  {"get_array_of_sets", getArrayOfSets, METH_NOARGS, ""},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPyArrayOfSetsModule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "testPyArrayOfSets",
  .m_doc = "",
  .m_size = -1,
  .m_methods = testPyArrayOfSetsFuncs,
};

END_ALLOW_DESIGNATED_INITIALIZERS

PyMODINIT_FUNC
PyInit_testPyArrayOfSets( void )
{
  LvArray::python::PyObjectRef<> module = PyModule_Create( &testPyArrayOfSetsModule );

  if( !LvArray::python::addPyLvArrayModule( module ) )
  {
    return nullptr;
  }

  return module.release();
}
