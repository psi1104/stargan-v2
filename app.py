import io
import os
import sys
import shutil
import uuid
import copy

import threading
import time
from queue import Empty, Queue

from flask import Flask, render_template, request, jsonify, send_file
from munch import Munch
from werkzeug.utils import secure_filename

from torch.backends import cudnn
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

from core.data_loader import get_test_loader
from core.solver import Solver
from core.wing import align_faces
from main import parse_args


if torch.cuda.is_available():
    pass
else:
    sys.exit('Cuda is not available')

#preload model
def create_model(args, model_type):
    args.resume_iter = 100000

    if model_type == 'CelebA-HQ':
        args.num_domains = 2
        args.w_hpf = 1
        args.checkpoint_dir = 'expr/checkpoints/celeba_hq'

    else:
        args.num_domains = 3
        args.w_hpf = 0
        args.checkpoint_dir = 'expr/checkpoints/afhq'

    model = Solver(args)
    return model

#update dir
def update_args(args, f_id):
    args.inp_dir = os.path.join(UPLOAD_FOLDER, f_id)
    args.out_dir = os.path.join(TARGET_FOLDER, f_id, f_id)
    args.src_dir = os.path.join(TARGET_FOLDER, f_id)
    args.result_dir = os.path.join(RESULT_FOLDER, f_id)

    return args

#remove image data
def remove_image(args):
    shutil.rmtree(args.inp_dir)
    shutil.rmtree(args.src_dir)
    shutil.rmtree(args.result_dir)

#detect face in image
def detect_face(im):
    sys.stderr.write("Detecting face using MTCNN face detector")
    try:
        bboxes, prob = face_detector.detect(im)
        w0, h0, w1, h1 = bboxes[0]
    except Exception as e:
        print(e)
        sys.stderr.write("Could not detect faces in the image")
        return False
    ###
    w_crop = (w1 - w0) / 3
    h_crop = (h1 - h0) / 3
    if w0 - w_crop > 0:
        w0 -= w_crop
    else:
        w0 = 0
    if w1 + w_crop < im.shape[1]:
        w1 += w_crop
    else:
        w1 = im.shape[1]

    if h0 - h_crop > 0:
        h0 -= h_crop
    else:
        h0 = 0
    if h1 + h_crop < im.shape[0]:
        h1 += h_crop
    else:
        h1 = im.shape[0]
    ###

    return im[int(h0):int(h1), int(w0):int(w1)]

#########################################################
UPLOAD_FOLDER = 'img_data/upload'
TARGET_FOLDER = 'img_data/target'
RESULT_FOLDER = 'img_data/result'

cudnn.benchmark = True

default_args = parse_args()
CelebA_HQ = create_model(copy.copy(default_args), 'CelebA-HQ')
AFHQ = create_model(copy.copy(default_args), 'AFHQ')

face_detector = MTCNN(select_largest=True, device=torch.device('cuda'))

requests_queue = Queue()
#########################################################
app = Flask(__name__, template_folder="./static/")
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

BATCH_SIZE=1
CHECK_INTERVAL=0.1

#run model
def run(input_file, model_type):
    f_id = str(uuid.uuid4())
    fname = secure_filename(input_file.filename)

    # save image to upload folder
    os.makedirs(os.path.join(UPLOAD_FOLDER, f_id), exist_ok=True)

    #update args
    args = update_args(default_args, f_id)
    torch.manual_seed(args.seed)

    #allocate solver and update args.ref_dir
    if model_type == "Human Face":
        solver = CelebA_HQ
        args.ref_dir = 'assets/representative/celeba_hq/ref'

        # human face crop
        pil_im = Image.open(input_file.stream).convert('RGB')
        im = np.uint8(pil_im)
        face_im = detect_face(copy.copy(im))

        # if can not detect face
        if type(face_im) == bool:
            return 'no face'

        Image.fromarray(face_im).save(os.path.join(UPLOAD_FOLDER, f_id, fname))
    else:
        solver = AFHQ
        args.ref_dir = 'assets/representative/afhq/ref'

        input_file.save(os.path.join(UPLOAD_FOLDER, f_id, fname))

    # align image
    align_faces(args, args.inp_dir, args.out_dir)

    #define loaders
    loaders = Munch(src=get_test_loader(root=args.src_dir,
                                    img_size=args.img_size,
                                    batch_size=args.val_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers),
                ref=get_test_loader(root=args.ref_dir,
                                    img_size=args.img_size,
                                    batch_size=args.val_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers))

    #generate image
    solver.sample(loaders, args.result_dir)

    #read image
    path = os.path.join(args.result_dir, 'reference.jpg')
    with open(path, 'rb') as f:
        data = f.read()
    result = io.BytesIO(data)

    #remove image data
    remove_image(args)

    return result

def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []

            while not (
              len(requests_batch) >= BATCH_SIZE # or
              #(len(requests_batch) > 0 #and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
            ):
              try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
              except Empty:
                continue

            batch_outputs = []

            for request in requests_batch:
                batch_outputs.append(run(request['input'][0], request['input'][1]))

            for request, output in zip(requests_batch, batch_outputs):
                request['output'] = output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

threading.Thread(target=handle_requests_by_batch).start()

@app.route('/predict', methods=['POST'])
def predict():
    print(requests_queue.qsize())
    if requests_queue.qsize() >= 1:
        return jsonify({'message': 'Too Many Requests'}), 429

    model_type = request.form['check_model']
    input_file = request.files['source']

    if input_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        return jsonify({'message': 'Only support jpeg, jpg or png'}), 400

    req = {
        'input': [input_file, model_type]
    }

    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    result = req['output']

    if result == 'no face':
        return jsonify({'message': 'Could not detect faces in the image'}), 400

    return send_file(result, mimetype='image/jpeg')

@app.route('/health')
def health():
    return "ok"

@app.route('/')
def main(): 
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, port=80, host='0.0.0.0')