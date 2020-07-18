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

/**
 * @file
 */

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "numpyHelpers.hpp"
#include "PySortedArray.hpp"
#include "pythonHelpers.hpp"
#include "../typeManipulation.hpp"

namespace LvArray
{
namespace python
{

namespace internal
{

/**
 * @brief Return @c true iff a Python exception has been set.
 * @return @c true iff a Python exception has been set.
 */
bool err();

/**
 * @brief Call the Python function @c func.
 * @param func The Python function to call.
 * @param args The arguments to call the function with, must be of length @c argc. These references are stolen and
 *   invalid after this call.
 * @param argc The number of arguments to call the function with.
 */
void callPyFunc( PyObject * func, PyObjectRef<> * args, long long const argc );

} // namespace internal

/**
 * @brief A C++ functor wrapper around a Python function.
 * @tparam ARGS A variadic parameter pack of types to call the function with.
 */
template< typename ... ARGS >
class PythonFunction
{
public:

  /**
   * @brief create a PythonFunction around @c pyfunc.
   * @param pyfunc the Python function to wrap, a new reference is created.
   */
  PythonFunction( PyObject * pyfunc ):
    m_function( pyfunc )
  { internal::xincref( pyfunc ); }

  /**
   * @brief Call the Python function with arguments @c args.
   * @param args The arguments to call the function with.
   * @note throws a PythonError (a C++ exception) if the args cannot be converted to
            Python objects or if the Python function raises an exception.
   */
  void operator()( ARGS ... args )
  {
    constexpr long long ARGC = sizeof ... (args);
    PyObjectRef<> pyArgs[ ARGC ];

    long long i = 0;
    typeManipulation::forEachArg( [&i, &pyArgs]( auto & arg )
    {
      pyArgs[ i ] = create( arg );
      ++i;
    }, args ... );

    for( i = 0; i < ARGC; ++i )
    {
      if( pyArgs[ i ] == nullptr )
      { throw PythonError(); }
    }

    // callPyFunc steals all of the pyArgs references
    internal::callPyFunc( m_function, pyArgs, ARGC );

    if( internal::err() )
    { throw PythonError(); }
  }

private:
  /// A reference to the wrapped python function.
  PyObjectRef<> m_function;
};


} // namespace python
} // namespace LvArray
