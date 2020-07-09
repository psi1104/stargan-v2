FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN pip install --upgrade pip
RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev ffmpeg

COPY requirements.txt .
#install python package
RUN pip install -r requirements.txt

COPY . .

CMD [ "bash" ]

# CMD [ "python", "./main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
#                --checkpoint_dir expr/checkpoints/celeba_hq \
#                --result_dir expr/results/celeba_hq \
#                --src_dir assets/representative/celeba_hq/src \
#                --ref_dir assets/representative/celeba_hq/ref" ]
