/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file umpireInterface.hpp
 * @brief Contains the LvArray umpire interface. This is only used to keep umpire/ResourceManager.hpp
 *   out of the includes for most headers.
 */

#pragma once

// TPL includes
#include <camp/resource.hpp>

namespace LvArray
{

namespace umpireInterface
{

/**
 * @brief Use memcpy to copy @p srcPointer in to @p dstPointer.
 * @param dstPointer The destination pointer.
 * @param srcPointer The source pointer.
 * @param size The number of bytes to copy.
 * @details If both @p src and @p dst were allocated with Umpire then the Umpire ResouceManager is used to perform
 *   the copy, otherwise std::memcpy is used.
 */
void copy( void * const dstPointer, void * const srcPointer, std::size_t const size );

/**
 * @brief Use memcpy to (maybe asynchronously) copy @p src in to @p dst.
 * @param dstPointer The destination slice.
 * @param srcPointer The source slice.
 * @param resource The resource to use.
 * @param size The number of bytes to copy.
 * @return The event corresponding to the copy.
 * @details If both @p src and @p dst were allocated with Umpire then the Umpire ResouceManager is used to perform
 *   the copy, otherwise std::memcpy is used. Umpire does not currently support asynchronous copying with host
 *   resources, in fact it does not even support synchronous copying with host resources. As such if a @p resource
 *   wraps a resource of type @c camp::resouces::Host the method that doesn't take a resource is used.
 */
camp::resources::Event copy( void * const dstPointer, void * const srcPointer,
                             camp::resources::Resource & resource, std::size_t const size );

/**
 * @brief Use memset to set the bytes of @p dstPointer.
 * @param dstPointer The destination pointer.
 * @param val The value (converted to a char) to set.
 * @param size The number of bytes to set.
 * @details If both @p src and @p dst were allocated with Umpire then the Umpire ResouceManager is used to perform
 *   the memset, otherwise std::memcpy is used.
 */
void memset( void * const dstPointer, int const val, std::size_t const size );

} // namespace umpireInterface
} // namespace LvArray
