set(MEMORYCHECK_COMMAND_OPTIONS "--tool=memcheck --partial-loads-ok=yes --error-limit=no --leak-check=full --show-reachable=yes --max-stackframe=16777216 --num-callers=20 --child-silent-after-fork=yes --track-origins=yes --error-exitcode=1 --errors-for-leak-kinds=all")

set(MEMORYCHECK_SUPPRESSIONS_FILE "${CMAKE_CURRENT_LIST_DIR}/scripts/valgrind.supp")
