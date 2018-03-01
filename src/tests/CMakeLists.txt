#
# Specify list of tests
#

set(newComponentTests
    testArrayAccesssor.cpp
    testManagedArray.cpp
    testObjectCatalog.cpp
    testSFINAE.cpp
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
                        DEPENDS_ON gtest cxx-utilities slic )

    blt_add_test( NAME ${test_name} 
                  COMMAND ${test_name}
                  )
                  
if( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
set_target_properties(${test_name} PROPERTIES COMPILE_FLAGS "-Wno-everything")
endif()
                  
endforeach()
