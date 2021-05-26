FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN chmod 777 /tmp

RUN apt-get update && apt-get -y install wget
COPY download.sh .

RUN bash download.sh pretrained-network-celeba-hq
RUN bash download.sh wing
RUN bash download.sh pretrained-network-afhq

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

ENV OPENCV_FOR_THREADS_NUM=1

EXPOSE 80

RUN pip install opencv-python-headless

CMD python app.py
