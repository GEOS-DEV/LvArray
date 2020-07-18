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
#include "pythonHelpers.hpp"
#include "../limits.hpp"

namespace LvArray
{
namespace python
{
namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void xincref( PyObject * const obj )
{ Py_XINCREF( obj ); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void xdecref( PyObject * const obj )
{ Py_XDECREF( obj ); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool isInstanceOf( PyObject * const obj, PyTypeObject * const type )
{
  PYTHON_ERROR_IF( obj == nullptr, PyExc_RuntimeError, "Passed a nullptr as an argument.", false );

  int isInstanceOfType = PyObject_IsInstance( obj, reinterpret_cast< PyObject * >( type ) );
  if( isInstanceOfType < 0 )
  {
    return false;
  }

  PYTHON_ERROR_IF( !isInstanceOfType, PyExc_TypeError, "Expect an argument of type " << type->tp_name, false );

  return true;
}

} // namespace internal

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool addTypeToModule( PyObject * const module,
                      PyTypeObject * const type,
                      char const * const typeName )
{
  if( PyType_Ready( type ) < 0 )
  {
    return false;
  }

  Py_INCREF( type );
  if( PyModule_AddObject( module, typeName, reinterpret_cast< PyObject * >( type ) ) < 0 )
  {
    Py_DECREF( type );
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::string const & value )
{ return PyUnicode_FromString( value.c_str() ); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * createPyListOfStrings( std::string const * const strptr, long long const size )
{
  PyObject * pylist = PyList_New( size );
  for( long long i = 0; i < size; ++i )
  {
    PyObject * pystr = PyUnicode_FromString( strptr[ i ].c_str() );
    PyList_SET_ITEM( pylist, integerConversion< Py_ssize_t >( i ), pystr );
  }
  return pylist;
}

} // namespace python
} // namespace LvArray
