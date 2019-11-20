/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file macros.hpp
 */

#pragma once

// Source includes
#include "stackTrace.hpp"

// System includes
#include <fstream>
#include <sstream>
#include <iostream>

#if defined(USE_CUDA)
  #include <cassert>
#endif

// This will interpret A as a string
#define STRINGIZE_NX( A ) #A

// This will macro expand A and then interpret that as a string.
#define STRINGIZE( A ) STRINGIZE_NX( A )

// Use this to mark an unused argument and silence compiler warnings
#define CXX_UTILS_UNUSED_ARG( X )

// Use this to mark an unused variable and silence compiler warnings.
#define CXX_UTILS_UNUSED_VARIABLE( X ) ( ( void ) X )

// Use this to mark a debug variable and silence compiler warnings
#define CXX_UTILS_DEBUG_VAR( X ) CXX_UTILS_UNUSED_VARIABLE( X )

#define LOCATION __FILE__ ":" STRINGIZE( __LINE__ )

#define VA_LIST( ... ) __VA_ARGS__

#define TYPEOFPTR( x ) typename std::remove_pointer< decltype(x) >::type
#define TYPEOFREF( x ) typename std::remove_reference< decltype(x) >::type


#define LVARRAY_LOG( msg ) \
  do \
  { \
    std::ostringstream oss; \
    oss << msg; \
    std::cout << oss.str() << std::endl; \
  } while( false )

#define LVARRAY_LOG_VAR( var ) LVARRAY_LOG( #var << " = " << var )

#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
    #define LVARRAY_ERROR_IF( EXP, msg ) assert( !(EXP) )
    #define LVARRAY_ERROR( msg ) assert( false )
    #define LVARRAY_ASSERT_MSG( EXP, msg ) assert( EXP )
    #define LVARRAY_ASSERT( EXP ) assert( EXP );
#endif

#if defined(__CUDA_ARCH__) && defined(NDEBUG)
  #define LVARRAY_ERROR_IF( EXP, msg ) if( EXP ) asm( "trap;" )
  #define LVARRAY_ERROR( msg ) LVARRAY_ERROR_IF( true, msg )
  #define LVARRAY_ASSERT_MSG( EXP, msg ) ((void) 0)
  #define LVARRAY_ASSERT( EXP ) ((void) 0)
#endif

#if !defined(__CUDA_ARCH__)
  #define LVARRAY_ERROR_IF( EXP, msg ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** ERROR" << std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << msg << std::endl; \
      cxx_utilities::handler1( EXIT_FAILURE ); \
    } \
  } while( false )

  #define LVARRAY_ERROR( msg ) LVARRAY_ERROR_IF( true, msg )

  #if !defined(NDEBUG)
    #define LVARRAY_ASSERT_MSG( EXP, msg ) LVARRAY_ERROR_IF( !(EXP), msg )
    #define LVARRAY_ASSERT( EXP ) LVARRAY_ASSERT_MSG( EXP, "" )
  #else
    #define LVARRAY_ASSERT_MSG( EXP, msg ) ((void) 0)
    #define LVARRAY_ASSERT( EXP ) ((void) 0)
  #endif

#endif

#define LVARRAY_WARNING_IF( EXP, msg ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** WARNING" << std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << msg << std::endl; \
    } \
  } while( false )

#define LVARRAY_WARNING( msg ) LVARRAY_WARNING_IF( true, msg )

#define LVARRAY_INFO_IF( EXP, msg ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** INFO "<<std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << msg << std::endl; \
    } \
  } while( false )

#define LVARRAY_INFO( msg ) LVARRAY_INFO_IF( true, msg )

#if !defined(NDEBUG)
  #define LVARRAY_CHECK( EXP, msg ) LVARRAY_WARNING_IF( !(EXP), msg )
#else
  #define LVARRAY_CHECK( EXP, msg ) ((void) 0)
#endif

#define LVARRAY_ERROR_IF_OP_MSG( lhs, OP, NOP, rhs, msg ) \
  LVARRAY_ERROR_IF( lhs OP rhs, \
                    "Expected " << #lhs << " " << #NOP << " " << #rhs << "\n" << \
                    "  " << #lhs << " = " << lhs << "\n" << \
                    "  " << #rhs << " = " << rhs << "\n" << \
                    msg )

#define LVARRAY_ERROR_IF_EQ_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, ==, !=, rhs, msg )
#define LVARRAY_ERROR_IF_EQ( lhs, rhs ) LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, "" )

#define LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, !=, ==, rhs, msg )
#define LVARRAY_ERROR_IF_NE( lhs, rhs ) LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, "" )

#define LVARRAY_ERROR_IF_GT_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, >, <=, rhs, msg )
#define LVARRAY_ERROR_IF_GT( lhs, rhs ) LVARRAY_ERROR_IF_GT_MSG( lhs, rhs, "" )

#define LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, >=, <, rhs, msg )
#define LVARRAY_ERROR_IF_GE( lhs, rhs ) LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, "" )

#define LVARRAY_ERROR_IF_LT_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, <, >=, rhs, msg )
#define LVARRAY_ERROR_IF_LT( lhs, rhs ) LVARRAY_ERROR_IF_GT_MSG( lhs, rhs, "" )

#define LVARRAY_ERROR_IF_LE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, <=, >, rhs, msg )
#define LVARRAY_ERROR_IF_LE( lhs, rhs ) LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, "" )


#define LVARRAY_ASSERT_OP_MSG( lhs, OP, rhs, msg ) \
  LVARRAY_ASSERT_MSG( lhs OP rhs, \
                      "  " << #lhs << " = " << lhs << "\n" << \
                      "  " << #rhs << " = " << rhs << "\n" << \
                      msg )

#define LVARRAY_ASSERT_EQ_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, ==, rhs, msg )
#define LVARRAY_ASSERT_EQ( lhs, rhs ) LVARRAY_ASSERT_EQ_MSG( lhs, rhs, "" )

#define LVARRAY_ASSERT_GT_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >, rhs, msg )
#define LVARRAY_ASSERT_GT( lhs, rhs ) LVARRAY_ASSERT_GT_MSG( lhs, rhs, "" )

#define LVARRAY_ASSERT_GE_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >=, rhs, msg )
#define LVARRAY_ASSERT_GE( lhs, rhs ) LVARRAY_ASSERT_GE_MSG( lhs, rhs, "" )
