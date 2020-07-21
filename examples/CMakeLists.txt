#
# Specify list of tests
#
set(exampleSources
    exampleTestFoo.cpp
    exampleBuffers.cpp
    exampleArray.cpp
    exampleSortedArray.cpp
    exampleArrayOfArrays.cpp
    exampleArrayOfSets.cpp
    exampleSparsityPatternAndCRSMatrix.cpp
    exampleTensorOps.cpp
   )

#
# Add gtest C++ based tests
#
foreach(example ${exampleSources})
    get_filename_component( example_name ${example} NAME_WE )
    blt_add_executable( NAME ${example_name}
                        SOURCES ${example}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest lvarray ${lvarray_dependencies} )

    target_include_directories( ${example_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

    blt_add_test( NAME ${example_name}
                  COMMAND ${example_name} )

endforeach()
