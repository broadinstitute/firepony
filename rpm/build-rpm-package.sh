#!/bin/sh

set -ex

CHROOT_PATH=$1
DISTRO=`basename $CHROOT_PATH`
FIREPONY_VERSION=`cat firepony/rpm/firepony.spec | grep Version: | awk '{print $2}'`

tar czvf $CHROOT_PATH/home/$USER/rpmbuild/SOURCES/firepony-${FIREPONY_VERSION}.tar.gz firepony
cp firepony/rpm/firepony.spec $CHROOT_PATH/home/$USER/rpmbuild/SPECS

set +e
sudo mount -t proc none $CHROOT_PATH/proc
sudo mount -t sysfs none $CHROOT_PATH/sys
sudo mount -o bind /dev $CHROOT_PATH/dev
sudo mount -t devpts devpts $CHROOT_PATH/dev/pts
set -e

sudo chroot $CHROOT_PATH su $USER -c "/usr/bin/rpmbuild -ba $HOME/rpmbuild/SPECS/firepony.spec"
sudo umount $CHROOT_PATH/proc
sudo umount $CHROOT_PATH/sys
sudo umount $CHROOT_PATH/dev/pts
sudo umount $CHROOT_PATH/dev

cp $CHROOT_PATH/home/$USER/rpmbuild/RPMS/x86_64/* packages.shadau.com/rpm/$DISTRO
cp $CHROOT_PATH/home/$USER/rpmbuild/SRPMS/* packages.shadau.com/rpm/$DISTRO

rpmsign --key-id=04F4CDB7 --addsign packages.shadau.com/rpm/$DISTRO/*.rpm

createrepo -u http://packages.shadau.com/rpm/$DISTRO packages.shadau.com/rpm/$DISTRO

