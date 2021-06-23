/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

namespace testFloatingPointExceptionsHelpers
{

/// Return @p x / @p y.
double divide( double const x, double const y );

/// Return @p x * @p y.
double multiply( double const x, double const y );

/// Performs an invalid floating point operations (generate a NaN).
double invalid();

} // namespace testFloatingPointExceptionsHelpers
