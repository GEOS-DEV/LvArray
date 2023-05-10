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
#include "python.hpp"
#include "../limits.hpp"
#include "pythonHelpers.hpp"

/// @cond DO_NOT_DOCUMENT

namespace LvArray
{
namespace python
{

/**
 *
 */
static PyMethodDef LvArrayFuncs[] = {
  { nullptr, nullptr, 0, nullptr }        /* Sentinel */
};

static constexpr char const * LvArrayDocString =
  "Manipulate LvArray objects from Python.\n\n"
  "This module provides classes for manipulating LvArray C++ classes from Python.\n"
  "Instances of the Python classes may not be constructed in Python;\n"
  "an existing LvArray object must be\n"
  "created in C++ and then converted to one of these Python classes."
;

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef LvArrayModuleFunctions = {
  PyModuleDef_HEAD_INIT,
  .m_name = "LvArray",
  .m_doc = LvArrayDocString,
  .m_size = -1,
  .m_methods = LvArrayFuncs
};

END_ALLOW_DESIGNATED_INITIALIZERS

static bool addConstants( PyObject * module )
{
  std::array< std::pair< long, char const * >, 3 > const constants = { {
    { static_cast< long >( LvArray::python::PyModify::READ_ONLY ), "READ_ONLY" },
    { static_cast< long >( LvArray::python::PyModify::MODIFIABLE ), "MODIFIABLE" },
    { static_cast< long >( LvArray::python::PyModify::RESIZEABLE ), "RESIZEABLE" }
  } };

  for( std::pair< long, char const * > const & pair : constants )
  {
    PYTHON_ERROR_IF( PyModule_AddIntConstant( module, pair.second, pair.first ), PyExc_RuntimeError,
                     "couldn't add constant", false );
  }

  PYTHON_ERROR_IF( PyModule_AddIntConstant( module, "CPU", static_cast< long >( LvArray::MemorySpace::host ) ), PyExc_RuntimeError,
                   "couldn't add constant", false );

  #if defined(LVARRAY_USE_CUDA)
  PYTHON_ERROR_IF( PyModule_AddIntConstant( module, "GPU", static_cast< long >( LvArray::MemorySpace::cuda ) ), PyExc_RuntimeError,
                   "couldn't add constant", false );
  #endif

  #if defined(LVARRAY_USE_HIP)
  PYTHON_ERROR_IF( PyModule_AddIntConstant( module, "GPU", static_cast< long >( LvArray::MemorySpace::hip ) ), PyExc_RuntimeError,
                   "couldn't add constant", false );
  #endif

  return true;
}

/**
 * @brief initialize the `pylvarray` module and add the Python types to the module.
 */
PyObjectRef<> getModule()
{
  // import_array();

  LvArray::python::PyObjectRef<> module{ PyModule_Create( &LvArrayModuleFunctions ) };
  if( module == nullptr )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayType(), "Array" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPySortedArrayType(), "SortedArray" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayOfArraysType(), "ArrayOfArrays" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayOfSetsType(), "ArrayOfSets" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyCRSMatrixType(), "CRSMatrix" ) )
  {
    return nullptr;
  }

  if( !addConstants( module ) )
  {
    return nullptr;
  }

  // Since we return module we don't want to decrease the reference count.
  return module.release();
}

/**
 * @brief add the `pylvarray` module, which defines all of the `pylvarray` classes, to a module.
 * @param module the Python module object to add `pylvarray` to as the attribute named "pylvarray".
 *  @c Equivalent to `import pylvarray; module.pylvarray = pylvarray`.
 * @return true if the import and addition succeeded, false otherwise.
 * @note It is recommended to invoke this function before using pylvarray's C++ API, e.g. in a
 *  @c module initialization function, since it ensures that all of the pylvarray types are defined.
 *  @c If the types are not defined but are attempted to be exported to python, it will trigger a segfault.
 */
bool addPyLvArrayModule( PyObject * module )
{
  LvArray::python::PyObjectRef<> pylvarrayModule { PyImport_ImportModule( "pylvarray" ) };
  if( PyModule_AddObject( module, "pylvarray", pylvarrayModule ) < 0 )
  {
    return false;
  }

  return true;
}

} // namespace python
} // namespace LvArray

PyMODINIT_FUNC
PyInit_pylvarray( void )
{ return LvArray::python::getModule().release(); }

/// @endcond DO_NOT_DOCUMENT
