// Source includes
#include "squareAll.hpp"
#include "../../src/jitti/jitti.hpp"

void squareAllOpenMP( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const dst,
                      LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const src,
                      std::string & policy )
{ squareAll< RAJA::omp_parallel_for_exec >( dst, src, policy ); }

jitti::SymbolTable exportedSymbols = {
  { std::string( "squareAllOpenMP" ),
     { reinterpret_cast< void * >( &squareAllOpenMP ), std::type_index( typeid( &squareAllOpenMP ) ) } }
};

extern "C"
{

jitti::SymbolTable const * getExportedSymbols()
{ return &exportedSymbols; }

} // extern "C"
