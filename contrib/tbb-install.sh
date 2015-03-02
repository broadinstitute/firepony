#!/bin/bash

set -e

tbb_SRC=$1
tbb_INSTALL=$2

echo "src: $tbb_SRC"
echo "install: $tbb_INSTALL"

# locate the _release build
cd $tbb_SRC
BUILD_DIRECTORY=`make info | egrep ^tbb_build_prefix= | sed s/^tbb_build_prefix=//`
BUILD_DIRECTORY+="_release"

TARGET_LIB_DIRECTORY="$tbb_INSTALL/lib"
TARGET_INC_DIRECTORY="$tbb_INSTALL/include"

# install binaries
mkdir -p $TARGET_LIB_DIRECTORY || true
cp -a $tbb_SRC/build/$BUILD_DIRECTORY/* $TARGET_LIB_DIRECTORY/

# create the static library
# xxxnsubtil: this is a gigantic hack and won't work in the general case, but works for firepony
ar cr $TARGET_LIB_DIRECTORY/libtbb.a $TARGET_LIB_DIRECTORY/*.o

# install headers
mkdir -p $TARGET_INC_DIRECTORY || true
cp -a $tbb_SRC/include/. $TARGET_INC_DIRECTORY/.

