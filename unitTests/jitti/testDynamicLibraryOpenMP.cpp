// Source includes
#include "squareAll.hpp"

extern "C"
void squareAllOpenMP( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const dst,
                      LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const src,
                      std::string & policy )
{ squareAll< RAJA::omp_parallel_for_exec >( dst, src, policy ); }