#
# Specify list of tests
#
set(testSources
    testArray.cpp
    testArrayOfArrays.cpp
    testArrayOfSets.cpp
    testArrayUtilities.cpp
    testArrayView.cpp
    testArraySlice.cpp
    testCRSMatrix.cpp
    testArrayHelpers.cpp
    testIntegerConversion.cpp
    testSortedArray.cpp
    testSortedArrayManipulation.cpp
    testSparsityPattern.cpp
    testStackArray.cpp
    testStringUtilities.cpp
    testPermutation.cpp
    testBuffers.cpp
    testSliceHelpers.cpp
   )

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

endforeach()

# Add testFloatingPointExceptions which has a dependency to prevent optimizations.
blt_add_executable( NAME testFloatingPointExceptions
                    SOURCES testFloatingPointExceptions.cpp testFloatingPointExceptionsHelpers.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

target_include_directories( testFloatingPointExceptions PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

blt_add_test( NAME testFloatingPointExceptions
              COMMAND testFloatingPointExceptions )

# Add testTensorOpsThreeSizes which has multiple translation units to speed up compilation.
blt_add_executable( NAME testTensorOps
                    SOURCES testTensorOps.cpp
                            testTensorOpsNoSize.cpp
                            testTensorOpsFixedSize.cpp
                            testTensorOpsOneSize.cpp
                            testTensorOpsTwoSizes1.cpp
                            testTensorOpsTwoSizes2.cpp
                            testTensorOpsTwoSizes3.cpp
                            testTensorOpsTwoSizes4.cpp
                            testTensorOpsThreeSizes1.cpp
                            testTensorOpsThreeSizes2.cpp
                            testTensorOpsThreeSizes3.cpp
                            testTensorOpsThreeSizes4.cpp
                            testTensorOpsThreeSizes5.cpp
                            testTensorOpsThreeSizes6.cpp
                            
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

target_include_directories( testTensorOps PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

blt_add_test( NAME testTensorOps
              COMMAND testTensorOps )
