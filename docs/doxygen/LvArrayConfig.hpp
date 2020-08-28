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

#define LVARRAY_BOUNDS_CHECK

#define LVARRAY_USE_CHAI

/* #undef LVARRAY_USE_CUDA */

#define LVARRAY_USE_MPI

/* #undef LVARRAY_USE_TOTALVIEW_OUTPUT */

#define LVARRAY_USE_OPENMP

#define LVARRAY_USE_CALIPER

#define LVARRAY_ADDR2LINE_EXEC /usr/bin/addr2line
