#if !defined(JITTI_TEMPLATE_HEADER_FILE)
  #error JITTI_TEMPLATE_HEADER_FILE must be defined!
#endif

#if !defined(JITTI_TEMPLATE_FUNCTION)
  #error JITTI_TEMPLATE_FUNCTION must be defined!
#endif

#if !defined(JITTI_TEMPLATE_PARAMS)
  #error JITTI_TEMPLATE_PARAMS must be defined!
#endif

#if !defined(JITTI_TEMPLATE_PARAMS_STRING)
  #error JITTI_TEMPLATE_PARAMS_STRING must be defined!
#endif

// Include the header the template is defined in.
#include JITTI_TEMPLATE_HEADER_FILE

#include <unordered_map>
#include <typeindex>
#include <string>

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

auto instantiation = JITTI_TEMPLATE_FUNCTION< JITTI_TEMPLATE_PARAMS >;

SymbolTable exportedSymbols = {
  { std::string( STRINGIZE( JITTI_TEMPLATE_FUNCTION ) "< " JITTI_TEMPLATE_PARAMS_STRING " >" ),
     { reinterpret_cast< void * >( instantiation ), std::type_index( typeid( instantiation ) ) } }
};

extern "C"
{

SymbolTable const * getExportedSymbols()
{ 
  return &exportedSymbols;
}

} // extern "C"

