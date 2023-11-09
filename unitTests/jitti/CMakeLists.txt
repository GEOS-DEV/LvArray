
target_include_directories( jitti PUBLIC ${CMAKE_BINARY_DIR}/include )

set( JITTI_TEST_OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}/lib )
set( JITTI_CXX_COMPILER ${CMAKE_CXX_COMPILER} )
set( JITTI_LINKER ${CMAKE_CXX_COMPILER} )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/jittiUnitTestConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/jittiUnitTestConfig.hpp )

set(testSources
    testUtils.cpp
    testCompilationInfo.cpp
   )

foreach(test ${testSources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest jitti )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src
                                                    ${CMAKE_BINARY_DIR}/include )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

    install( TARGETS ${test_name}
             DESTINATION bin )

endforeach()

# Logic for generating the header containing the command used to build testFunction.
set( linkDirectories "" )
set( linkLibraries "" )

if( ENABLE_CUDA )
    list( APPEND linkDirectories ${CUDA_TOOLKIT_ROOT_DIR}/lib64 )
    list( APPEND linkLibraries cudart_static cudadevrt )
endif()

add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/testFunctionCompileCommands.hpp
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/testFunction.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/testFunctionCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${linkDirectories}
                                   --linkLibraries ${linkLibraries}
                                   --includeDirectories ${CMAKE_BINARY_DIR}/include )

set(testSourcesDependentOnCompileCommands
    testFunction.cpp
    testCache.cpp
    )

foreach(test ${testSourcesDependentOnCompileCommands})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test} ${CMAKE_BINARY_DIR}/include/testFunctionCompileCommands.hpp
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest jitti cuda )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src
                                                    ${CMAKE_BINARY_DIR}/include )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

    install( TARGETS ${test_name}
             DESTINATION bin )

endforeach()
