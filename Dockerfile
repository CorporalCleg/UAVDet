FROM nvidia/cuda:12.4.0-devel-ubuntu20.04



#nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


WORKDIR /UAVDet




RUN apt update
RUN apt install python3-pip -y
RUN pip3 install "ultralytics == 8.2.32"
RUN pip3 install "streamlit == 1.35.0"
RUN pip3 install "pillow-heif == 0.16.0"


ENV TZ=UTC
RUN date

ENV TZ="America/New_York"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && apt-get install ffmpeg -y locales




RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev  \
        libgles2-mesa-dev
        


COPY uav_detector ./uav_detector
COPY scripts ./


CMD ["bash"]
