ARG BASE_OS_IMAGE
FROM $BASE_OS_IMAGE

MAINTAINER wangyin.yx

ADD functions /etc/rc.d/init.d/functions

RUN echo "%sdev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    groupadd sdev

RUN dnf install -y \
        unzip wget which findutils rsync tar \
        gcc gcc-c++ libstdc++-static gdb coreutils \
        binutils bash glibc-devel libdb glibc glibc-langpack-en bison lld && \
    dnf clean all

RUN dnf install -y \
        emacs-nox git git-lfs openblas-devel \
        https://mirrors.aliyun.com/docker-ce/linux/centos/8/aarch64/stable/Packages/docker-ce-cli-26.1.3-1.el8.aarch64.rpm \
    	nfs-utils && \
    dnf clean all

RUN dnf install -y \
    	java-latest-openjdk-devel.aarch64 && \
    dnf clean all

ARG CONDA_URL
RUN wget $CONDA_URL -O /tmp/conda.sh && \
    sh /tmp/conda.sh -b -p /opt/conda310/ && \
    rm /tmp/conda.sh -f

ARG BAZEL_URL
RUN wget $BAZEL_URL -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

RUN git config --system core.hooksPath .githooks && \
    git lfs install

ARG PYPI_URL
ADD deps /tmp/deps
RUN /opt/conda310/bin/pip install -r /tmp/deps/requirements_torch_cpu.txt -r /tmp/deps/requirements_base.txt -i $PYPI_URL && \
    rm -rf /tmp/deps && /opt/conda310/bin/pip cache purge

ARG BAZELISK_URL
RUN wget $BAZELISK_URL -O /usr/local/bin/bazelisk && chmod a+x /usr/local/bin/bazelisk

RUN wget https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_24.10/arm-performance-libraries_24.10_rpm_gcc.tar \
    && tar -xvf arm-performance-libraries_24.10_rpm_gcc.tar \
    && sh ./arm-performance-libraries_24.10_rpm/arm-performance-libraries_24.10_rpm.sh -a \
    && rm -rf arm-performance-libraries_24.10_rpm arm-performance-libraries_24.10_rpm_gcc.tar

ENV LD_PRELOAD=/opt/arm/armpl_24.10_gcc/lib/libarmpl_mp.so
