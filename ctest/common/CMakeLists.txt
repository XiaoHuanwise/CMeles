# Common utility tests (Log, Timer).

add_executable(test_log test_log.cpp)
target_link_libraries(test_log PRIVATE cmeles_core)
add_test(NAME log COMMAND test_log)

add_executable(test_timer test_timer.cpp)
target_link_libraries(test_timer PRIVATE cmeles_core)
add_test(NAME timer COMMAND test_timer)
