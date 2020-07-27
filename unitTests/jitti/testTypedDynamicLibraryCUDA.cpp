// Source includes
#include "squareAll.hpp"
#include "../../src/jitti/jitti.hpp"

void squareAllCUDA( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const & dst,
                    LvArray::ArrayView< int const, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const & src,
                    std::string & policy )
{ squareAll< RAJA::cuda_exec< 32 > >( dst, src, policy ); }

jitti::SymbolTable exportedSymbols = {
  { std::string( "squareAllCUDA" ),
     { reinterpret_cast< void * >( &squareAllCUDA ), std::type_index( typeid( &squareAllCUDA ) ) } }
};

extern "C"
{

jitti::SymbolTable const * getExportedSymbols()
{ return &exportedSymbols; }

} // extern "C"
