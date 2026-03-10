#!/bin/bash
set -e
gcc -o tests/run_tests tests/test_distance.c -lm
./tests/run_tests
rm tests/run_tests
