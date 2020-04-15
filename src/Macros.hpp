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
 * @file Macros.hpp
 */

#pragma once

/// Source includes
#include "stackTrace.hpp"

/// System includes
#include <fstream>
#include <sstream>
#include <iostream>
#include <type_traits>

#if defined(USE_CUDA)
  #include <cassert>
#endif

/**
 * @brief Convert @p A into a string.
 * @param A the token to convert to a string.
 */
#define STRINGIZE_NX( A ) #A

/**
 * @brief Convert the macro expansion of @p A into a string.
 * @param A the token to convert to a string.
 */
#define STRINGIZE( A ) STRINGIZE_NX( A )

/**
 * @brief Mark @p X as an unused argument, used to silence compiler warnings.
 * @param X the unused argument.
 */
#define LVARRAY_UNUSED_ARG( X )

/**
 * @brief Mark @p X as an unused variable, used to silence compiler warnings.
 * @param X the unused variable.
 */
#define LVARRAY_UNUSED_VARIABLE( X ) ( ( void ) X )

/**
 * @brief Mark @p X as an debug variable, used to silence compiler warnings.
 * @param X the debug variable.
 */
#define LVARRAY_DEBUG_VAR( X ) LVARRAY_UNUSED_VARIABLE( X )

/// Expands to a string representing the current file and line.
#define LOCATION __FILE__ ":" STRINGIZE( __LINE__ )

/**
 * @brief Given an expression @p X that evaluates to a pointer, expands to the type pointed to.
 * @param X The expression to evaluate.
 */
#define TYPEOFPTR( X ) std::remove_pointer_t< decltype( X ) >

/**
 * @brief Given an expression @p X that evaluates to a reference, expands to the type referred to.
 * @param X The expression to evaluate.
 */
#define TYPEOFREF( X ) std::remove_reference_t< decltype( X ) >

/**
 * @brief Print the expression @p MSG.
 * @param MSG The expression to print, may be anything that can be streamed to std::out.
 */
#define LVARRAY_LOG( MSG ) std::cout << MSG << std::endl

/**
 * @brief Print the name of the variable @p VAR along with its value.
 * @param VAR The variable to print.
 */
#define LVARRAY_LOG_VAR( VAR ) LVARRAY_LOG( #VAR << " = " << VAR )

/**
 * @brief Abort execution if @p EXP is true.
 * @param EXP The expression to check.
 * @param MSG The message to associate with the error, can be anything streamable to std::cout.
 * @note This macro can be used in both host and device code.
 * @note Tries to provide as much information about the location of the error
 *       as possible. On host this should result in the file and line of the error
 *       and a stack trace along with the provided message. On device none of this is
 *       guaranteed. In fact it is only guaranteed to abort the current kernel.
 */
#if defined(__CUDA_ARCH__)
  #if !defined(NDEBUG)
    #define LVARRAY_ERROR_IF( EXP, MSG ) assert( !(EXP) )
  #else
    #define LVARRAY_ERROR_IF( EXP, MSG ) if( EXP ) asm ( "trap;" )
  #endif
#else
  #define LVARRAY_ERROR_IF( EXP, MSG ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** ERROR" << std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << "***** Controlling expression (should be false): " << STRINGIZE( EXP ) << std::endl; \
      std::cout << MSG << std::endl; \
      cxx_utilities::handler1( EXIT_FAILURE ); \
    } \
  } while( false )
#endif

/**
 * @brief Abort execution.
 * @param MSG The message to associate with the error, can be anything streamable to std::cout.
 */
#define LVARRAY_ERROR( MSG ) LVARRAY_ERROR_IF( true, MSG )

/**
 * @brief Abort execution if @p EXP is false but only when
 *        NDEBUG is not defined..
 * @param EXP The expression to check.
 * @param MSG The message to associate with the error, can be anything streamable to std::cout.
 * @note This macro can be used in both host and device code.
 * @note Tries to provide as much information about the location of the error
 *       as possible. On host this should result in the file and line of the error
 *       and a stack trace along with the provided message. On device none of this is
 *       guaranteed. In fact it is only guaranteed to abort the current kernel.
 */
#if !defined(NDEBUG)
  #define LVARRAY_ASSERT_MSG( EXP, MSG ) LVARRAY_ERROR_IF( !(EXP), MSG )
#else
  #define LVARRAY_ASSERT_MSG( EXP, MSG ) ((void) 0)
#endif

/// Assert @p EXP is true with no message.
#define LVARRAY_ASSERT( EXP ) LVARRAY_ASSERT_MSG( EXP, "" )

/**
 * @brief Print a warning if @p EXP is true.
 * @param EXP The expression to check.
 * @param MSG The message to associate with the warning, can be anything streamable to std::cout.
 */
#define LVARRAY_WARNING_IF( EXP, MSG ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** WARNING" << std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << "***** Controlling expression (should be false): " << STRINGIZE( EXP ) << std::endl; \
      std::cout << MSG << std::endl; \
    } \
  } while( false )

/// Print a warning with the message @p MSG.
#define LVARRAY_WARNING( MSG ) LVARRAY_WARNING_IF( true, MSG )

#define LVARRAY_INFO_IF( EXP, msg ) \
  do \
  { \
    if( EXP ) \
    { \
      std::cout << "***** INFO " << std::endl; \
      std::cout << "***** LOCATION: " << LOCATION << std::endl; \
      std::cout << "***** Controlling expression: " << STRINGIZE( EXP ) << std::endl; \
      std::cout << msg << std::endl; \
    } \
  } while( false )

#define LVARRAY_INFO( msg ) LVARRAY_INFO_IF( true, msg )

#define LVARRAY_ERROR_IF_OP_MSG( lhs, OP, NOP, rhs, msg ) \
  LVARRAY_ERROR_IF( lhs OP rhs, \
                    msg << "\n" << \
                    "Expected " << #lhs << " " << #NOP << " " << #rhs << "\n" << \
                    "  " << #lhs << " = " << lhs << "\n" << \
                    "  " << #rhs << " = " << rhs << "\n" )

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
                      msg << "\n" << \
                      "  " << #lhs << " = " << lhs << "\n" << \
                      "  " << #rhs << " = " << rhs << "\n" )

#define LVARRAY_ASSERT_EQ_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, ==, rhs, msg )
#define LVARRAY_ASSERT_EQ( lhs, rhs ) LVARRAY_ASSERT_EQ_MSG( lhs, rhs, "" )

#define LVARRAY_ASSERT_NE_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, !=, rhs, msg )
#define LVARRAY_ASSERT_NE( lhs, rhs ) LVARRAY_ASSERT_NE_MSG( lhs, rhs, "" )

#define LVARRAY_ASSERT_GT_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >, rhs, msg )
#define LVARRAY_ASSERT_GT( lhs, rhs ) LVARRAY_ASSERT_GT_MSG( lhs, rhs, "" )

#define LVARRAY_ASSERT_GE_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >=, rhs, msg )
#define LVARRAY_ASSERT_GE( lhs, rhs ) LVARRAY_ASSERT_GE_MSG( lhs, rhs, "" )



#if defined(USE_CUDA) && defined(__CUDACC__)
  #define LVARRAY_HOST_DEVICE __host__ __device__
  #define LVARRY_DEVICE __device__

// This pragma disables nvcc warnings about calling a host function from a host-device
// function. This is used on templated host-device functions where some template instantiations
// call host only code. This is safe as long as the host only instantiations are only called on
// the host. Furthermore it seems like trying to call a host only instantiation on the device leads
// to other compiler errors/warnings.
// To use place directly above a function declaration.
  #define DISABLE_HD_WARNING _Pragma("hd_warning_disable")
#else
  #define LVARRAY_HOST_DEVICE
  #define LVARRY_DEVICE
  #define DISABLE_HD_WARNING
#endif


#if defined(__clang__)
  #define LVARRAY_RESTRICT __restrict__
  #define LVARRAY_RESTRICT_THIS
  #define CONSTEXPRFUNC constexpr
#elif defined(__GNUC__)
  #if defined(__INTEL_COMPILER)
    #define LVARRAY_RESTRICT __restrict__
    #define LVARRAY_RESTRICT_THIS
    #define CONSTEXPRFUNC
  #else
    #define LVARRAY_RESTRICT __restrict__
    #define LVARRAY_RESTRICT_THIS
    #define CONSTEXPRFUNC constexpr
  #endif
#endif
