#pragma once
#include <umpire/ResourceManager.hpp>

void getUmpireAllocatorName( T const * const ptr )
{
  return umpire::ResourceManager::getInstance().getAllocator( ptr ).getName();
}
