#
# Specify list of tests
#

set(testSources
    testArray.cpp
    testArrayOfArrays.cpp
    testArrayOfSets.cpp
    testArrayUtilities.cpp
    testArrayView.cpp
    testCRSMatrix.cpp
    testFloatingPointExceptions.cpp
    testArrayHelpers.cpp
    testIntegerConversion.cpp
    testSortedArray.cpp
    testSortedArrayManipulation.cpp
    testSparsityPattern.cpp
    testStackArray.cpp
    testStringUtilities.cpp
    testPermutation.cpp
    testBuffers.cpp
   )

message( "cxx_dependencies = ${cxx_dependencies}" )

#
# Add gtest C++ based tests
#
foreach(test ${testSources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test} testHelperFunctions.cpp
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest cxx-utilities ${cxx_dependencies} )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}
                  )

endforeach()
