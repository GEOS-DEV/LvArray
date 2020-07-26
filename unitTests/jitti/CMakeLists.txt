#
# Logic for testDynamicLibrary
#
blt_add_library( NAME testDynamicLibrarySerial
                 SOURCES testDynamicLibrarySerial.cpp
                 OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                 DEPENDS_ON lvarray ${lvarray_dependencies}
                 SHARED ON )

set_property( TARGET testDynamicLibrarySerial PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set( testDynamicLibraryDependencies testDynamicLibrarySerial )

if( ENABLE_OPENMP )
  blt_add_library( NAME testDynamicLibraryOpenMP
                   SOURCES testDynamicLibraryOpenMP.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies}
                   SHARED ON )

  set_property( TARGET testDynamicLibraryOpenMP PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testDynamicLibraryDependencies testDynamicLibraryOpenMP )
endif()

if( ENABLE_CUDA )
  blt_add_library( NAME testDynamicLibraryCUDA
                   SOURCES testDynamicLibraryCUDA.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies}
                   SHARED ON )

  set_property( TARGET testDynamicLibraryCUDA PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testDynamicLibraryDependencies testDynamicLibraryCUDA )
endif()

blt_add_executable( NAME testDynamicLibrary
                    SOURCES testDynamicLibrary.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray jitti ${lvarray_dependencies} gtest )

add_dependencies( testDynamicLibrary ${testDynamicLibraryDependencies} )

blt_add_test( NAME testDynamicLibrary
              COMMAND testDynamicLibrary )

#
# Logic for testTypedDynamicLibrary
#
blt_add_library( NAME testTypedDynamicLibrarySerial
                 SOURCES testTypedDynamicLibrarySerial.cpp
                 OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                 DEPENDS_ON lvarray ${lvarray_dependencies} jitti
                 SHARED ON )

set_property( TARGET testTypedDynamicLibrarySerial PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
set( testTypedDynamicLibraryDependencies testTypedDynamicLibrarySerial )

if( ENABLE_OPENMP )
  blt_add_library( NAME testTypedDynamicLibraryOpenMP
                   SOURCES testTypedDynamicLibraryOpenMP.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies} jitti
                   SHARED ON )

  set_property( TARGET testTypedDynamicLibraryOpenMP PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testTypedDynamicLibraryDependencies testTypedDynamicLibraryOpenMP )
endif()

if( ENABLE_CUDA )
  blt_add_library( NAME testTypedDynamicLibraryCUDA
                   SOURCES testTypedDynamicLibraryCUDA.cpp
                   OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti
                   DEPENDS_ON lvarray ${lvarray_dependencies} jitti
                   SHARED ON )

  set_property( TARGET testTypedDynamicLibraryCUDA PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
  list( APPEND testTypedDynamicLibraryDependencies testTypedDynamicLibraryCUDA )
endif()

blt_add_executable( NAME testTypedDynamicLibrary
                    SOURCES testTypedDynamicLibrary.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON lvarray ${lvarray_dependencies} jitti gtest )

add_dependencies( testTypedDynamicLibrary ${testTypedDynamicLibraryDependencies} )

blt_add_test( NAME testTypedDynamicLibrary
              COMMAND testTypedDynamicLibrary )

#
# Logic for testTemplateCompiler
#
blt_add_library( NAME squareAllJIT
                 SOURCES squareAllJIT.cpp
                 HEADERS squareAllJIT.hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                 DEPENDS_ON lvarray jitti ${lvarray_dependencies}
                 SHARED ON )

get_target_property( squareAll_LINK_LIBRARIES squareAllJIT LINK_LIBRARIES )

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
                                   --cpp ${CMAKE_CURRENT_LIST_DIR}/squareAllJIT.cpp
                                   --hpp ${CMAKE_BINARY_DIR}/include/squareAllCompileCommands.hpp
                                   --linker ${CMAKE_CXX_COMPILER}
                                   --linkDirectories ${linkDirectories}
                                   --linkLibraries '${squareAll_LINK_LIBRARIES}' )

blt_add_executable( NAME testTemplateCompiler
                    SOURCES testTemplateCompiler.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAllJIT gtest ${lvarray_dependencies} )

blt_add_test( NAME testTemplateCompiler
              COMMAND testTemplateCompiler )
