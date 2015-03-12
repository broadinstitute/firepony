#!/bin/bash

set -e

# for reasons unknown to me, some of the required boost header libraries never
# get installed by the boost build system; this script basically does that
# after the fact 

boost_SRC=$1
boost_INSTALL=$2

for f in $boost_SRC/src/boost/libs/*/include
do
    cd $f
    tar cf - . | (cd $boost_INSTALL/include && tar xvf -)
done

