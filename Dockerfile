FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN pip install --upgrade pip
RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev ffmpeg

COPY requirements.txt .

#install python package
RUN pip install -r requirements.txt

COPY . .

ENV OPENCV_FOR_THREADS_NUM=1

EXPOSE 80

CMD python app.py