/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "LvArrayConfig.hpp"
#include "umpireInterface.hpp"

// TPL includes
#if defined( LVARRAY_USE_UMPIRE )
#include <umpire/ResourceManager.hpp>
#endif

// System includes
#include <cstring>

namespace LvArray
{

namespace umpireInterface
{

void copy( void * const dstPointer, void * const srcPointer, std::size_t const size )
{
#if defined( LVARRAY_USE_UMPIRE )
  umpire::ResourceManager & rm = umpire::ResourceManager::getInstance();
  if( rm.hasAllocator( dstPointer ) && rm.hasAllocator( srcPointer ) )
  {
    rm.copy( dstPointer, srcPointer, size );
    return;
  }
#endif

  std::memcpy( dstPointer, srcPointer, size );
}

camp::resources::Event copy( void * const dstPointer, void * const srcPointer,
                             camp::resources::Resource & resource, std::size_t const size )
{
#if defined( LVARRAY_USE_UMPIRE )
  umpire::ResourceManager & rm = umpire::ResourceManager::getInstance();

  bool const allocatedByUmpire = rm.hasAllocator( dstPointer ) && rm.hasAllocator( srcPointer );

  // Umpire breaks if you supply a host resource.
  if( allocatedByUmpire && resource.try_get< camp::resources::Host >() )
  {
    rm.copy( dstPointer, srcPointer, size );
    return resource.get_event();
  }

  if( allocatedByUmpire )
  {
    return rm.copy( dstPointer, srcPointer, resource, size );
  }
#endif

  std::memcpy( dstPointer, srcPointer, size );
  return resource.get_event();
}

void memset( void * const dstPointer, int const val, std::size_t const size )
{
#if defined( LVARRAY_USE_UMPIRE )
  umpire::ResourceManager & rm = umpire::ResourceManager::getInstance();
  if( rm.hasAllocator( dstPointer ) )
  {
    return rm.memset( dstPointer, val, size );
  }
#endif

  std::memset( dstPointer, val, size );
}

} // namespace umpireInterface
} // namespace LvArray
