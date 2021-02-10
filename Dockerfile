FROM nvidia/cuda:10.1-cudnn8-runtime

RUN chmod 777 /tmp
RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev ffmpeg wget
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update
RUN apt install python3.8 -y
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py
RUN python get-pip.py
RUN ln -sf /usr/bin/pip3.8 /usr/bin/pip

COPY download.sh .

RUN bash download.sh pretrained-network-celeba-hq
RUN bash download.sh wing
RUN bash download.sh pretrained-network-afhq

COPY requirements.txt .

#install python package
RUN pip install -r requirements.txt

COPY . .

ENV OPENCV_FOR_THREADS_NUM=1

EXPOSE 80

CMD python app.py