#!/bin/bash

set -ex

if [ -z "$1" ]
then
	echo "usage: $0 <distribution> <extraversion>"
	exit 1
fi

DISTRIBUTION=$1
EXTRAVERSION=$2

pushd firepony
git reset --hard HEAD
popd

FIREPONY_FULLVERSION=`head -1 firepony/debian/changelog|awk '{print $2}'|tr -d '(' | tr -d ')'`
FIREPONY_VERSION=`head -1 firepony/debian/changelog|awk '{print $2}'|tr -d '(' | tr -d ')' | sed 's/\-.*$//'`
DSC_NAME=firepony_$FIREPONY_FULLVERSION${DISTRIBUTION}${EXTRAVERSION}.dsc

TARGET_BINARY_PATH=dists/$DISTRIBUTION/main/binary-amd64
TARGET_SOURCE_PATH=dists/$DISTRIBUTION/main/source

echo "dist: $DISTRIBUTION ($TARGET_BINARY_PATH)"
echo "extra: $EXTRAVERSION"
echo "firepony: $FIREPONY_FULLVERSION $FIREPONY_VERSION $DSC_NAME"

#pbuilder-dist $DISTRIBUTION update

if [ ! -d $TARGET_BINARY_PATH ]
then
	mkdir -p $TARGET_BINARY_PATH
fi

if [ ! -d $TARGET_SOURCE_PATH ]
then
	mkdir -p $TARGET_SOURCE_PATH
fi

pushd firepony
debian/filter-changelog.py debian/changelog $DISTRIBUTION $EXTRAVERSION > debian/changelog.tmp
mv debian/changelog.tmp debian/changelog
popd

if [ ! -f "firepony_$FIREPONY_VERSION.orig.tar.gz" ]
then
	tar --exclude=firepony/debian -czf firepony_$FIREPONY_VERSION.orig.tar.gz firepony
fi

pushd firepony
debuild -S
popd

rm -f $HOME/pbuilder/${DISTRIBUTION}_result/*
pbuilder-dist $DISTRIBUTION build $DSC_NAME --debbuildopts "-j6"

cp $HOME/pbuilder/${DISTRIBUTION}_result/* $TARGET_BINARY_PATH
mv firepony_*$DISTRIBUTION* $TARGET_SOURCE_PATH
cp firepony_$FIREPONY_VERSION.orig.tar.gz $TARGET_SOURCE_PATH

#pushd $TARGET_BINARY_PATH
#dpkg-scanpackages . > Packages
#popd
#
#pushd $TARGET_SOURCE_PATH
#dpkg-scansources . > Sources
#popd

pushd firepony
git checkout debian/changelog
popd

reprepro includedeb ${DISTRIBUTION} $TARGET_BINARY_PATH/firepony_${FIREPONY_FULLVERSION}${DISTRIBUTION}${EXTRAVERSION}_amd64.deb
reprepro includedsc ${DISTRIBUTION} $TARGET_SOURCE_PATH/firepony_${FIREPONY_FULLVERSION}${DISTRIBUTION}${EXTRAVERSION}.dsc

