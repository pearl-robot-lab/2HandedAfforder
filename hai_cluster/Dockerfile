FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-f66cbce
# For CPU, use:
# FROM determinedai/environments:py-3.10-pytorch-2.0-cpu-748dda4

RUN apt-get update --allow-releaseinfo-change
RUN apt-get update && apt-get -y install build-essential cmake diffutils findutils gcc g++ gdb gcp grep gzip p7zip-full p7zip unzip rsync autotools-dev bzip2 unrar htop lz4
RUN apt-get update && apt-get -y install tar wget screen tmux nano emacs vim sshfs
RUN apt-get update && apt-get -y install iputils-ping iputils-tracepath ibverbs-utils infiniband-diags libibverbs-dev perftest
# for opencv
RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6

RUN echo 'root:root' | chpasswd
RUN echo 'PermitRootLogin prohibit-password' >> /etc/ssh/sshd_config