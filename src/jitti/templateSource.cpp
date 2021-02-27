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

#include "types.hpp"

auto instantiation = JITTI_TEMPLATE_FUNCTION< JITTI_TEMPLATE_PARAMS >;

jitti::SymbolTable exportedSymbols = {
  { std::string( STRINGIZE( JITTI_TEMPLATE_FUNCTION ) "< " JITTI_TEMPLATE_PARAMS_STRING " >" ),
     { reinterpret_cast< void * >( instantiation ), std::type_index( typeid( instantiation ) ) } }
};

extern "C"
{

jitti::SymbolTable const * getExportedSymbols()
{ 
  return &exportedSymbols;
}

} // extern "C"

