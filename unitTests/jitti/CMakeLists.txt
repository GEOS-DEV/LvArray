# # Used to get the compilation flags for squareAll
blt_add_library( NAME squareAllDummy
                 SOURCES squareAll.cpp
                 SHARED TRUE
                 DEPENDS_ON lvarray jitti cuda )

target_include_directories( squareAllDummy PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src )

blt_add_library( NAME squareAll
                 SOURCES squareAll.cpp
                 HEADERS squareAll.hpp
                 DEPENDS_ON lvarray jitti cuda )

blt_add_executable( NAME testJIT
                    SOURCES testJIT.cpp
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                    DEPENDS_ON squareAll gtest cuda )

target_include_directories( testJIT PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src )

blt_add_test( NAME testJIT
              COMMAND testJIT )
