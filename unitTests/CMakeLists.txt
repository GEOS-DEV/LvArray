###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

#
# Specify list of tests
#
set( testSources
     testArray1D.cpp
     testArray1DOfArray1D.cpp
     testArray1DOfArray1DOfArray1D.cpp
     testArrayOfArrays.cpp
     testArrayOfSets.cpp
     testArraySlice.cpp
     testArrayView_copyAssignmentOperator.cpp
     testArrayView_copyConstructor.cpp
     testArrayView_defaultConstructor.cpp
     testArrayView_emptyMove.cpp
     testArrayView_modifyInKernel.cpp
     testArrayView_modifyInMultipleKernels.cpp
     testArrayView_move.cpp
     testArrayView_moveMultipleTimes.cpp
     testArrayView_moveNoTouch.cpp
     testArrayView_readInKernel.cpp
     testArrayView_setValues.cpp
     testArrayView_setValuesFromView.cpp
     testArrayView_toSlice.cpp
     testArrayView_toSliceConst.cpp
     testArrayView_toView.cpp
     testArrayView_toViewConst.cpp
     testArrayView_typeConversion.cpp
     testArrayView_udcToSlice.cpp
     testArrayView_udcToSliceConst.cpp
     testArrayView_udcToViewConst.cpp
     testArrayView_zero.cpp
     testArray_clear.cpp
     testArray_copyAssignmentOperator.cpp
     testArray_copyConstructor.cpp
     testArray_defaultConstructor.cpp
     testArray_indexing.cpp
     testArray_moveAssignmentOperator.cpp
     testArray_moveConstructor.cpp
     testArray_reserveAndCapacity.cpp
     testArray_resize.cpp
     testArray_resizeDefault.cpp
     testArray_resizeDimension.cpp
     testArray_resizeFromArgs.cpp
     testArray_resizeFromPointer.cpp
     testArray_resizeWithoutInitializationOrDestruction.cpp
     testArray_sizedConstructor.cpp
     testArray_toView.cpp
     testArray_toViewConst.cpp
     testBuffers.cpp
     testCRSMatrix.cpp
     testIndexing.cpp
     testInput.cpp
     testIntegerConversion.cpp
     testInvalidOperations.cpp
     testMath.cpp
     testMemcpy.cpp
     testPrettyPrinters.cpp
     testSliceHelpers.cpp
     testSortedArray.cpp
     testSortedArrayManipulation.cpp
     testSparsityPattern.cpp
     testStackArray.cpp
     testStackTrace.cpp
     testTensorOpsDeterminant.cpp
     testTensorOpsEigen.cpp
     testTensorOpsInverseOneArg.cpp
     testTensorOpsInverseTwoArgs.cpp
     testTensorOpsSymDeterminant.cpp
     testTensorOpsSymInverseOneArg.cpp
     testTensorOpsSymInverseTwoArgs.cpp
     testTypeManipulation.cpp
   )

blt_list_append( TO testSources ELEMENTS testChaiBuffer.cpp testArray_ChaiBuffer.cpp IF ENABLE_CHAI )

#
# Add gtest C++ based tests
#
foreach(test ${testSources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

    install(TARGETS ${test_name}
            DESTINATION bin)

endforeach()

# Add testFloatingPointExceptions which has a dependency to prevent optimizations.
blt_add_executable( NAME testFloatingPointExceptions
                    SOURCES testFloatingPointExceptions.cpp testFloatingPointExceptionsHelpers.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

# Need to avoid optimization to catch invalid operations
if( APPLE AND ${CMAKE_CXX_COMPILER} MATCHES "clang" )
  set_source_files_properties( testFloatingPointExceptionsHelpers.cpp PROPERTIES COMPILE_FLAGS "-O0" )
endif()

target_include_directories( testFloatingPointExceptions PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

blt_add_test( NAME testFloatingPointExceptions
              COMMAND testFloatingPointExceptions )

install(TARGETS testFloatingPointExceptions
        DESTINATION bin)

# Add testTensorOps which has multiple translation units to speed up compilation.
blt_add_executable( NAME testTensorOps
                    SOURCES testTensorOps.cpp
                            testTensorOpsFixedSize.cpp
                            testTensorOpsNoSize.cpp
                            testTensorOpsOneSize.cpp
                            testTensorOpsThreeSizes1.cpp
                            testTensorOpsThreeSizes2.cpp
                            testTensorOpsThreeSizes3.cpp
                            testTensorOpsThreeSizes4.cpp
                            testTensorOpsThreeSizes5.cpp
                            testTensorOpsThreeSizes6.cpp
                            testTensorOpsTwoSizes1.cpp
                            testTensorOpsTwoSizes2.cpp
                            testTensorOpsTwoSizes3.cpp
                            testTensorOpsTwoSizes4.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

target_include_directories( testTensorOps PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

blt_add_test( NAME testTensorOps
              COMMAND testTensorOps )

install(TARGETS testTensorOps
        DESTINATION bin)

if( ENABLE_PYLVARRAY )
    add_subdirectory( python )
endif()
