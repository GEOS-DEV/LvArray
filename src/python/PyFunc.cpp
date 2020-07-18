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
#include "PyFunc.hpp"

namespace LvArray
{
namespace python
{

namespace internal
{

bool err()
{ return PyErr_Occurred() != nullptr; }

void callPyFunc( PyObject * func, PyObjectRef<> * args, long long const argc )
{
  Py_ssize_t const argc_ssize = static_cast< Py_ssize_t >( argc );
  PyObjectRef<> tup{ PyTuple_New( argc_ssize ) };
  if( tup == nullptr )
  {
    return;
  }

  for( Py_ssize_t i = 0; i < argc_ssize; ++i )
  {
    // The references get stolen by PyObject_CallObject.
    PyTuple_SET_ITEM( tup.get(), i, args[ i ].release() );
  }

  Py_XDECREF( PyObject_CallObject( func, tup.get() ) );
}

} // namespace internal

} // namespace python
} // namespace LvArray
