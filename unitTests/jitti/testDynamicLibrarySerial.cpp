// Source includes
#include "squareAll.hpp"

extern "C"
void squareAllSerial( LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const dst,
                      LvArray::ArrayView< int, 1, 0, std::ptrdiff_t, LvArray::NewChaiBuffer > const src,
                      std::string & policy )
{ squareAll< RAJA::loop_exec >( dst, src, policy ); }

