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
 * @file python.hpp
 * @brief Includes all the other python related headers.
 */

#pragma once

// source includes
#include "PyArray.hpp"
#include "PySortedArray.hpp"
#include "PyArrayOfArrays.hpp"
#include "PyArrayOfSets.hpp"
#include "PyCRSMatrix.hpp"
#include "PyFunc.hpp"
#include "../typeManipulation.hpp"

namespace LvArray
{

/**
 * @brief Contains all the Python code.
 */
namespace python
{

/**
 * @brief add the `pylvarray` module, which defines all of the `pylvarray` classes, to a module.
 * @param module the Python module object to add `pylvarray` to as the attribute named "pylvarray".
 *   Equivalent to `import pylvarray; module.pylvarray = pylvarray`.
 * @return true if the import and addition succeeded, false otherwise.
 * @note It is recommended to invoke this function before using pylvarray's C++ API, e.g. in a
 *   module initialization function, since it ensures that all of the pylvarray types are defined.
 *   If the types are not defined but are attempted to be exported to python, it will trigger a segfault.
 */
bool addPyLvArrayModule( PyObject * module );

/**
 * @brief Expands to a static constexpr template bool @code CanCreate< T > @endcode which is true iff
 *   @c T is a type which LvArray can export to Python.
 */
IS_VALID_EXPRESSION( CanCreate, T, LvArray::python::create( std::declval< T & >() ) );

} // namespace python
} // namespace LvArray
