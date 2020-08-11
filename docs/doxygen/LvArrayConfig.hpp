/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file LvArrayConfig.hpp
 * @brief Contains platform specific parameters defined by CMake.
 */

#pragma once

#define USE_ARRAY_BOUNDS_CHECK

#ifndef USE_CHAI
#define USE_CHAI
#endif

#ifndef USE_CUDA
#define USE_CUDA
#endif

#ifndef USE_MPI
#define USE_MPI
#endif

#ifndef USE_TOTALVIEW_OUTPUT
#define USE_TOTALVIEW_OUTPUT
#endif

#ifndef USE_OPENMP
#define USE_OPENMP
#endif

#ifndef USE_CALIPER
#define USE_CALIPER
#endif

#define LVARRAY_ADDR2LINE_EXEC "/usr/bin/addr2line"
