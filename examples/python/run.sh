#!/bin/bash
#Arguments: ACCELERATOR_FLAG EXAMPLE_FOLDER

COMPILER_SCRIPT_DPCPP=/opt/intel/oneapi/setvars.sh
COMPILER_SCRIPT_CUDA=/opt/oneAPICUDA/startup.sh

cd "$2" || exit

if [ "$1" = "CUDA" ]
then
  . ${COMPILER_SCRIPT_CUDA}
else
  . ${COMPILER_SCRIPT_DPCPP}
fi

mkdir -p build
cd build/ || exit

cmake ../../../../ -DACCELERATOR_FLAG="$1" -DCMAKE_BUILD_TYPE=Release
make pydphdc

cd ../

python main.py "$1"