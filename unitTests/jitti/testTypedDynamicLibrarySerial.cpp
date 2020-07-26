// Source includes
#include "squareAll.hpp"
#include "../../src/jitti/jitti.hpp"

void squareAllSerial( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const dst,
                      LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const src,
                      std::string & policy )
{ squareAll< RAJA::loop_exec >( dst, src, policy ); }

jitti::SymbolTable exportedSymbols = {
  { std::string( "squareAllSerial" ),
     { reinterpret_cast< void * >( &squareAllSerial ), std::type_index( typeid( &squareAllSerial ) ) } }
};

extern "C"
{

jitti::SymbolTable const * getExportedSymbols()
{ return &exportedSymbols; }

} // extern "C"
