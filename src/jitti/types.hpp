#pragma once

// System includes
#include <string>
#include <unordered_map>
#include <utility>
#include <typeindex>

namespace jitti
{

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

} // namespace jitti
