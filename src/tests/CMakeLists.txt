#
# Specify list of tests
#

set(newComponentTests
    testArrayAccesssor.cpp
    testObjectCatalog.cpp
    testStringUtilities.cpp
   )

#
# Add gtest C++ based tests
#
foreach(test ${newComponentTests})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        SYSTEM_DEPENDS_ON gtest
                        DEPENDS_ON cxx-utilities )

    blt_add_test( NAME ${test_name} 
                  COMMAND ${test_name}
                  )
endforeach()
