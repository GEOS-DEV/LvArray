// Source includes
#include "squareAll.hpp"

extern "C"
void squareAllCUDA( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const dst,
                    LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const src,
                    std::string & policy )
{ squareAll< RAJA::cuda_exec< 32 > >( dst, src, policy ); }
