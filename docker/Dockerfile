FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade && apt-get install -y vim wget gcc make git zlib1g-dev libssl-dev sudo libgl1 libglib2.0-0

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python python \
    && pip3 install --upgrade pip

RUN pip3 install --upgrade pip

COPY ./requirements.txt /home/requirements.txt

RUN pip install -r /home/requirements.txt
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/@331ecdd5306104614cb414b16fbcd9d1a8d40e1e
