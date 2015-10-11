Summary:            Quality score recalibrator for DNA sequencing data
Name:               firepony
Version:            1.0.1
Release:            21%{?dist}
License:            BSD
Group:              Engineering and Scientific
Source:             https://github.com/broadinstitute/firepony/archive/firepony-%{?version}.tar.gz
URL:                https://github.com/broadinstitute/firepony
BuildRequires:      cuda-minimal-build-7-0
BuildRequires:      cuda-misc-headers-7-0
BuildRequires:      gcc
BuildRequires:      gcc-c++
BuildRequires:      git
BuildRequires:      cmake
BuildRequires:      zlib
BuildRequires:      zlib-devel
BuildRequires:      make

%description
Firepony is a base quality score recalibration tool for aligned NGS read data.
It is a reimplementation of the GATK Base Quality Score Recalibrator algorithm
in C++ with optional NVIDIA GPU support.

The output of Firepony is compatible with and can be used by GATK directly in a
pre-existing NGS processing pipeline.

%prep
%setup -n firepony

%build
env CFLAGS="" CXXFLAGS="" cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=/usr \
    -DINCLUDE_INSTALL_DIR:PATH=/usr/include \
    -DLIB_INSTALL_DIR:PATH=/usr/lib64 \
    -DSYSCONF_INSTALL_DIR:PATH=/etc \
    -DSHARE_INSTALL_PREFIX:PATH=/usr/share \
    -DLIB_SUFFIX=64 \
    .

make VERBOSE=1 %{?_smp_mflags}

%install
%make_install

%files
/usr/bin/firepony
/usr/bin/firepony-loader

%changelog
* Sun Oct 10 2015 Nuno Subtil <subtil at gmail.com> - 1.0.1
- BAQ phase performance improvements

* Tue Jul 14 2015 Nuno Subtil <subtil at gmail.com> - 1.0.0
- No changes from previous release

* Sun Jul 12 2015 Nuno Subtil <subtil at gmail.com> - 0.9.9
- Initial build of RPM package

