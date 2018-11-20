#
# Specify list of tests
#

set(newComponentTests
    testArray.cpp
    testIntegerConversion.cpp
    testObjectCatalog.cpp
    testSFINAE.cpp
    testStringUtilities.cpp
    testChaiVector.cpp
    testSortedArray.cpp
    testStackArray.cpp
   )

if( ${ENABLE_CUDA})
    set( cxx_dependencies ${cxx_dependencies} cuda )
endif()

#
# Add gtest C++ based tests
#
foreach(test ${newComponentTests})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest cxx-utilities ${cxx_dependencies} )

    blt_add_test( NAME ${test_name} 
                  COMMAND ${test_name}
                  )

    if( ( NOT ${ENABLE_CUDA} ) AND ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" ) )    
        set_target_properties(${test_name} PROPERTIES COMPILE_FLAGS "-Wno-everything")    
    endif()

endforeach()


    # blt_add_executable( NAME arrayAccessorPerf
    #                     SOURCES arrayAccessorPerformance.cpp arrayAccessorPerformanceMain.cpp
    #                     INCLUDES arrayAccessorPerformance.hpp
    #                     OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON gtest cxx-utilities ${CXX_UTILITIES_DEP} )

    # blt_add_executable( NAME add
    #                     SOURCES add.cu 
    #                     INCLUDES 
    #                     OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON cxx-utilities ${CXX_UTILITIES_DEP} )
                        