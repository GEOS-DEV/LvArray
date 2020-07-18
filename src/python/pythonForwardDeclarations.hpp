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
 * @file pythonForwardDeclarations.hpp
 * @brief Forward declarations of Python Objects.
 * @note Taken from https://mail.python.org/pipermail/python-dev/2003-August/037601.html
 */

#pragma once

/// @cond DO_NOT_DOCUMENT

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;

struct _typeobject;
typedef _typeobject PyTypeObject;
#endif

/// @endcond DO_NOT_DOCUMENT
