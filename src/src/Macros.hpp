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

/*
 * macros.hpp
 */

#pragma once

// Use this to mark an unused argument and silence compiler warnings
#define CXX_UTILS_UNUSED_ARG( X )

// Use this to mark an unused variable and silence compiler warnings.
#define CXX_UTILS_UNUSED_VARIABLE( X ) ( ( void ) X )

// Use this to mark a debug variable and silence compiler warnings
#define CXX_UTILS_DEBUG_VAR( X ) CXX_UTILS_UNUSED_VARIABLE( X )

#define LOCATION __FILE__ ":" STRINGIZE( __LINE__ )

#define VA_LIST( ... ) __VA_ARGS__

#define TYPEOFPTR( x ) typename std::remove_pointer<decltype(x)>::type
#define TYPEOFREF( x ) typename std::remove_reference<decltype(x)>::type
