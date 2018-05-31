// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-746361. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the GEOSX Simulation Framework.

//
// GEOSX is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
/*
 * macros.hpp
 *
 *  Created on: Nov 13, 2014
 *      Author: rrsettgast
 */

#ifndef MACROS_HPP_
#define MACROS_HPP_



#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define LOCATION __FILE__ ":" STRINGIZE(__LINE__)

#define VA_LIST(...) __VA_ARGS__


#endif /* MACROS_HPP_ */


//#if __cplusplus == 199711L // There is no value for 03 vs 98.
//#define CXX_STD 03
//#elif __cplusplus == 201103L
//#define CXX_STD 11
//#define USE_CXX11
//#elif __cplusplus == 201402L
//#define CXX_STD 14
//#define USE_CXX11
//#elif __cplusplus > 201402L
//#define CXX_STD 1z
//#define USE_CXX11
//#elif
//#error "No allowable value of __cplusplus preprocessor flag is available.
// __cplusplus must be >= 199711L"
//#endif
