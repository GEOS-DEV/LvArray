#if !defined(JITTI_TEMPLATE_HEADER_FILE)
  #error JITTI_TEMPLATE_HEADER_FILE must be defined!
#endif

#if !defined(JITTI_TEMPLATE_FUNCTION)
  #error JITTI_TEMPLATE_FUNCTION must be defined!
#endif

#if !defined(JITTI_TEMPLATE_PARAMS)
  #error JITTI_TEMPLATE_PARAMS must be defined!
#endif

// Include the header the template is defined in
#include JITTI_TEMPLATE_HEADER_FILE

// Include the jitti header
#include "jitti.hpp"


auto instantiation = JITTI_TEMPLATE_FUNCTION< JITTI_TEMPLATE_PARAMS >;

extern "C"
{

jitti::SymbolTable exportedSymbols = {
  { std::string( STRINGIZE( JITTI_TEMPLATE_FUNCTION ) "< " STRINGIZE( JITTI_TEMPLATE_PARAMS ) " >" ),
    jitti::createEntry( instantiation ) }
};

} // extern "C"

