#!/bin/bash
find ./include -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i
find ./src -name '*.cpp' -o -name '*.hpp'  -o -name '*.h'| xargs clang-format -i