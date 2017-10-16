/*
 * FloatingPointEnvironment.h
 *
 *  Created on: Jun 14, 2016
 *      Author: settgast1
 */

#ifndef COMPONENTS_CORE_SRC_CODINGUTILITIES_SETSIGNALHANDLING_HPP_
#define COMPONENTS_CORE_SRC_CODINGUTILITIES_SETSIGNALHANDLING_HPP_

#ifndef __APPLE__
#define __APPLE__ 0
#endif
#include <exception>

namespace cxx_utilities
{

void setSignalHandling(  void (*handler)( int ) );


} /* namespace geosx */

#endif /* COMPONENTS_CORE_SRC_CODINGUTILITIES_SETSIGNALHANDLING_HPP_ */
