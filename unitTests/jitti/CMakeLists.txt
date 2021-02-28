#
# Specify list of tests
#
set(testSources
    testUtils.cpp
    testCompilationInfo.cpp
    testFunction.cpp
   )

#
# Add gtest C++ based tests
#
foreach(test ${testSources})
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON gtest jitti )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../src )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name} )

    install(TARGETS ${test_name}
            DESTINATION bin)

endforeach()

# Logic for testTemplateCompiler
# The first step is to compile the library that will hold the compilation info. It doesn't have to be in a
# separate library from the test executable but it makes the dependency on the generated header cleaner.
blt_add_library( NAME squareAllJIT
                 SOURCES squareAllJIT.cpp
                 HEADERS squareAllJIT.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 DEPENDS_ON lvarray jitti ${lvarray_dependencies}
                 SHARED ON )

# The second set is to create the generated header. It parses the output compile_commands.json file to get the
# command used to build squareAllJIT.cpp. It adapts this to compile the dynamic library. However we need to manually
# specify the link commands. As with above we don't want to link to anything except if we're using cuda.
set( linkDirectories "" )
set( linkLibraries "" )

if( ENABLE_CUDA )
    list( APPEND linkDirectories ${CUDA_TOOLKIT_ROOT_DIR}/lib64 )
    list( APPEND linkLibraries cudart_static cudadevrt )
endif()

add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/squareAllJIT.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${linkDirectories}
                                   --linkLibraries ${linkLibraries}
                                   --includeDirectories ${CMAKE_BINARY_DIR}/include )

blt_add_executable( NAME testTemplateCompiler
                    SOURCES testTemplateCompiler.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAllJIT gtest ${lvarray_dependencies} )

blt_add_test( NAME testTemplateCompiler
              COMMAND testTemplateCompiler )
