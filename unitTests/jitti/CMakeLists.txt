#
# Logic for testDynamicLibrary
#
blt_add_library( NAME testDynamicLibrarySerial
                 SOURCES testDynamicLibrarySerial.cpp
                 OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                 DEPENDS_ON lvarray ${lvarray_dependencies}
                 SHARED ON )

set( testDynamicLibraryDependencies testDynamicLibrarySerial )

if( ENABLE_OPENMP )
  blt_add_library( NAME testDynamicLibraryOpenMP
                   SOURCES testDynamicLibraryOpenMP.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies}
                   SHARED ON )

  list( APPEND testDynamicLibraryDependencies testDynamicLibraryOpenMP )
endif()

if( ENABLE_CUDA )
  blt_add_library( NAME testDynamicLibraryCUDA
                   SOURCES testDynamicLibraryCUDA.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies}
                   SHARED ON )

  list( APPEND testDynamicLibraryDependencies testDynamicLibraryCUDA )

  # blt_print_target_properties(TARGET testDynamicLibraryCUDA)
endif()


blt_add_executable( NAME testDynamicLibrary
                    SOURCES testDynamicLibrary.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray jitti gtest ${lvarray_dependencies} )

add_dependencies( testDynamicLibrary ${testDynamicLibraryDependencies} )

# blt_print_target_properties(TARGET testDynamicLibrary)

blt_add_test( NAME testDynamicLibrary
              COMMAND testDynamicLibrary )

#
# Logic for testTemplateCompiler
#
blt_add_library( NAME squareAll
                 SOURCES squareAll.cpp 
                 HEADERS squareAll.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 DEPENDS_ON lvarray jitti ${lvarray_dependencies}
                 SHARED ON )

get_target_property( squareAll_LINK_LIBRARIES squareAll LINK_LIBRARIES )

set( linkDirectories ${CMAKE_BINARY_DIR}/lib )

if( ENABLE_CALIPER )
    list(APPEND linkDirectories ${CALIPER_DIR}/lib64 )
endif()

if( ENABLE_CHAI )
    list(APPEND linkDirectories ${CHAI_DIR}/lib )
endif()

add_custom_command( OUTPUT ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../../src/jitti/generateCompileCommandsHeader.py
                                   ${CMAKE_BINARY_DIR}/compile_commands.json
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/squareAll.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${linkDirectories}
                                   --linkLibraries '${squareAll_LINK_LIBRARIES}' )

# blt_print_target_properties(TARGET squareAll)

blt_add_executable( NAME testTemplateCompiler
                    SOURCES testTemplateCompiler.cpp squareAll.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAll gtest ${lvarray_dependencies} )

blt_add_test( NAME testTemplateCompiler
              COMMAND testTemplateCompiler )
