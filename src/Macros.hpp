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
#include "LvArrayConfig.hpp"
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
 * @brief Print the expression.
 */
#define LVARRAY_LOG( ... ) std::cout << __VA_ARGS__ << std::endl

/**
 * @brief Print the expression string along with its value.
 */
#define LVARRAY_LOG_VAR( ... ) LVARRAY_LOG( STRINGIZE( __VA_ARGS__ ) << " = " << __VA_ARGS__ )

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
      std::ostringstream __oss; \
      __oss << "***** ERROR\n"; \
      __oss << "***** LOCATION: " LOCATION "\n"; \
      __oss << "***** Controlling expression (should be false): " STRINGIZE( EXP ) "\n"; \
      __oss << MSG << "\n"; \
      __oss << LvArray::stackTrace(); \
      std::cout << __oss.str() << std::endl; \
      LvArray::abort(); \
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
      std::ostringstream __oss; \
      __oss << "***** WARNING\n"; \
      __oss << "***** LOCATION: " LOCATION "\n"; \
      __oss << "***** Controlling expression (should be false): " STRINGIZE( EXP ) "\n"; \
      __oss << MSG; \
      std::cout << __oss.str() << std::endl; \
    } \
  } while( false )

/**
 * @brief Print a warning with a message.
 * @param MSG The message to print.
 */
#define LVARRAY_WARNING( MSG ) LVARRAY_WARNING_IF( true, MSG )

/**
 * @brief Print @p msg along with the location if @p EXP is true.
 * @param EXP The expression to test.
 * @param MSG The message to print.
 */
#define LVARRAY_INFO_IF( EXP, MSG ) \
  do \
  { \
    if( EXP ) \
    { \
      std::ostringstream __oss; \
      __oss << "***** INFO\n"; \
      __oss << "***** LOCATION: " LOCATION "\n"; \
      __oss << "***** Controlling expression: " STRINGIZE( EXP ) "\n"; \
      __oss << MSG; \
      std::cout << __oss.str() << std::endl; \
    } \
  } while( false )

/**
 * @brief Print @p msg along with the location.
 * @param msg The message to print.
 */
#define LVARRAY_INFO( msg ) LVARRAY_INFO_IF( true, msg )

/**
 * @brief Abort execution if @p lhs @p OP @p rhs.
 * @param lhs The left side of the operation.
 * @param OP The operation to apply.
 * @param NOP The opposite of @p OP, used in the message.
 * @param rhs The right side of the operation.
 * @param msg The message to diplay.
 */
#define LVARRAY_ERROR_IF_OP_MSG( lhs, OP, NOP, rhs, msg ) \
  LVARRAY_ERROR_IF( lhs OP rhs, \
                    msg << "\n" << \
                    "Expected " << #lhs << " " << #NOP << " " << #rhs << "\n" << \
                    "  " << #lhs << " = " << lhs << "\n" << \
                    "  " << #rhs << " = " << rhs << "\n" )

/**
 * @brief Raise a hard error if two values are equal.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_EQ_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, ==, !=, rhs, msg )

/**
 * @brief Raise a hard error if two values are equal.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_EQ( lhs, rhs ) LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, "" )

/**
 * @brief Raise a hard error if two values are not equal.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, !=, ==, rhs, msg )

/**
 * @brief Raise a hard error if two values are not equal.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_NE( lhs, rhs ) LVARRAY_ERROR_IF_NE_MSG( lhs, rhs, "" )

/**
 * @brief Raise a hard error if one value compares greater than the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_GT_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, >, <=, rhs, msg )

/**
 * @brief Raise a hard error if one value compares greater than the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_GT( lhs, rhs ) LVARRAY_ERROR_IF_GT_MSG( lhs, rhs, "" )

/**
 * @brief Raise a hard error if one value compares greater than or equal to the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, >=, <, rhs, msg )

/**
 * @brief Raise a hard error if one value compares greater than or equal to the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_GE( lhs, rhs ) LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, "" )

/**
 * @brief Raise a hard error if one value compares less than the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_LT_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, <, >=, rhs, msg )

/**
 * @brief Raise a hard error if one value compares less than the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_LT( lhs, rhs ) LVARRAY_ERROR_IF_LT_MSG( lhs, rhs, "" )

/**
 * @brief Raise a hard error if one value compares less than or equal to the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ERROR_IF_LE_MSG( lhs, rhs, msg ) LVARRAY_ERROR_IF_OP_MSG( lhs, <=, >, rhs, msg )

/**
 * @brief Raise a hard error if one value compares less than or equal to the other.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ERROR_IF_LE( lhs, rhs ) LVARRAY_ERROR_IF_GE_MSG( lhs, rhs, "" )

/**
 * @brief Abort execution if @p lhs @p OP @p rhs is false.
 * @param lhs The left side of the operation.
 * @param OP The operation to apply.
 * @param rhs The right side of the operation.
 * @param msg The message to diplay.
 */
#define LVARRAY_ASSERT_OP_MSG( lhs, OP, rhs, msg ) \
  LVARRAY_ASSERT_MSG( lhs OP rhs, \
                      msg << "\n" << \
                      "  " << #lhs << " = " << lhs << "\n" << \
                      "  " << #rhs << " = " << rhs << "\n" )

/**
 * @brief Assert that two values compare equal in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ASSERT_EQ_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, ==, rhs, msg )

/**
 * @brief Assert that two values compare equal in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ASSERT_EQ( lhs, rhs ) LVARRAY_ASSERT_EQ_MSG( lhs, rhs, "" )

/**
 * @brief Assert that two values compare not equal in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ASSERT_NE_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, !=, rhs, msg )

/**
 * @brief Assert that two values compare not equal in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ASSERT_NE( lhs, rhs ) LVARRAY_ASSERT_NE_MSG( lhs, rhs, "" )

/**
 * @brief Assert that one value compares greater than the other in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ASSERT_GT_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >, rhs, msg )

/**
 * @brief Assert that one value compares greater than the other in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ASSERT_GT( lhs, rhs ) LVARRAY_ASSERT_GT_MSG( lhs, rhs, "" )

/**
 * @brief Assert that one value compares greater than or equal to the other in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 * @param msg a message to log (any expression that can be stream inserted)
 */
#define LVARRAY_ASSERT_GE_MSG( lhs, rhs, msg ) LVARRAY_ASSERT_OP_MSG( lhs, >=, rhs, msg )

/**
 * @brief Assert that one value compares greater than or equal to the other in debug builds.
 * @param lhs expression to be evaluated and used as left-hand side in comparison
 * @param rhs expression to be evaluated and used as right-hand side in comparison
 */
#define LVARRAY_ASSERT_GE( lhs, rhs ) LVARRAY_ASSERT_GE_MSG( lhs, rhs, "" )

#if defined(USE_CUDA) && defined(__CUDACC__)
/// Mark a function for both host and device usage.
#define LVARRAY_HOST_DEVICE __host__ __device__

/// Mark a function for only device usage.
#define LVARRAY_DEVICE __device__

/**
 * @brief Disable host device warnings.
 * @details This pragma disables nvcc warnings about calling a host function from a host-device
 *   function. This is used on templated host-device functions where some template instantiations
 *   call host only code. This is safe as long as the host only instantiations are only called on
 *   the host. To use place directly above a the template.
 */
#define DISABLE_HD_WARNING _Pragma("hd_warning_disable")
#else
/// Mark a function for both host and device usage.
#define LVARRAY_HOST_DEVICE

/// Mark a function for only device usage.
#define LVARRAY_DEVICE

/**
 * @brief Disable host device warnings.
 * @details This pragma disables nvcc warnings about calling a host function from a host-device
 *   function. This is used on templated host-device functions where some template instantiations
 *   call host only code. This is safe as long as the host only instantiations are only called on
 *   the host. To use place directly above a the template.
 */
#define DISABLE_HD_WARNING
#endif


#if defined(__clang__)
  #define LVARRAY_RESTRICT __restrict__
  #define LVARRAY_RESTRICT_REF __restrict__
  #define LVARRAY_RESTRICT_THIS
#elif defined(__GNUC__)
  #if defined(__INTEL_COMPILER)
    #define LVARRAY_RESTRICT __restrict__
    #define LVARRAY_RESTRICT_REF __restrict__
    #define LVARRAY_RESTRICT_THIS
  #else
    #define LVARRAY_RESTRICT __restrict__
    #define LVARRAY_RESTRICT_REF __restrict__
    #define LVARRAY_RESTRICT_THIS
  #endif
#endif

#if !defined(USE_ARRAY_BOUNDS_CHECK)
#define CONSTEXPR_WITHOUT_BOUNDS_CHECK constexpr
#else
#define CONSTEXPR_WITHOUT_BOUNDS_CHECK
#endif

#if defined(NDEBUG)
#define CONSTEXPR_WITH_NDEBUG constexpr
#else
#define CONSTEXPR_WITH_NDEBUG
#endif

#if !defined(USE_ARRAY_BOUNDS_CHECK)
/**
 * @brief Expands to constexpr when array bound checking is disabled.
 */
#define CONSTEXPR_WITHOUT_BOUNDS_CHECK constexpr
#else
/**
 * @brief Expands to constexpr when array bound checking is disabled.
 */
#define CONSTEXPR_WITHOUT_BOUNDS_CHECK
#endif

#if defined(NDEBUG)
/**
 * @brief Expands to constexpr in release builds (when NDEBUG is defined).
 */
#define CONSTEXPR_WITH_NDEBUG constexpr
#else
/**
 * @brief Expands to constexpr in release builds (when NDEBUG is defined).
 */
#define CONSTEXPR_WITH_NDEBUG
#endif
