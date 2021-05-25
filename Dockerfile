FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN chmod 777 /tmp
# RUN apt update
# RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev ffmpeg wget
# RUN apt install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa

# RUN python3 -v
# RUN apt update
# RUN apt install python3.8 -y
# RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
# RUN python get-pip.py
# RUN ln -sf /usr/bin/pip3.8 /usr/bin/pip

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN apt-get -y install wget
COPY download.sh .

RUN bash download.sh pretrained-network-celeba-hq
RUN bash download.sh wing
RUN bash download.sh pretrained-network-afhq

COPY requirements.txt .

# RUN pip3 install --upgrade pip3
RUN pip3 install --upgrade pip

#install python package
RUN pip3 install -r requirements.txt

COPY . .

ENV OPENCV_FOR_THREADS_NUM=1

EXPOSE 80

RUN apt-get install -y libgl1-mesa-glx

CMD python3 app.py
