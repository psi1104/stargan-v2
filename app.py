import os
import shutil
import uuid
import copy

from flask import Flask, render_template, request, jsonify, send_from_directory
from munch import Munch
from werkzeug.utils import secure_filename

from torch.backends import cudnn
import torch
from core.data_loader import get_test_loader
from core.solver import Solver
from core.wing import align_faces
from main import parse_args

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

#########################################################
UPLOAD_FOLDER = 'img_data/upload'
TARGET_FOLDER = 'img_data/target'
RESULT_FOLDER = 'img_data/result'

cudnn.benchmark = True

default_args = parse_args()
CelebA_HQ = create_model(copy.copy(default_args), 'CelebA-HQ')
AFHQ = create_model(copy.copy(default_args), 'AFHQ')

#########################################################
app = Flask(__name__, template_folder="./static/")

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['check_model']
    input_file = request.files['source']
    if not input_file:
        return jsonify({'message': 'nofile'}), 400
    if input_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        return jsonify({'message': 'only support jpeg, jpg or png'}), 400

    f_id = str(uuid.uuid4())
    fname = secure_filename(input_file.filename)

    #save image to upload folder
    os.makedirs(os.path.join(UPLOAD_FOLDER, f_id), exist_ok=True)
    input_file.save(os.path.join(UPLOAD_FOLDER, f_id, fname))

    #update args
    args = update_args(default_args, f_id)
    torch.manual_seed(args.seed)

    #align image
    align_faces(args, args.inp_dir, args.out_dir)

    #allocate solver and update args.ref_dir
    if model_type == "Human Face":
        solver = CelebA_HQ
        args.ref_dir = 'assets/representative/celeba_hq/ref'
    else:
        solver = AFHQ
        args.ref_dir = 'assets/representative/afhq/ref'

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

    result = send_from_directory(directory=args.result_dir, filename='reference.jpg')

    #remove image data
    remove_image(args)

    return result


@app.route('/health')
def health():
    return "ok"

@app.route('/')
def main(): 
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, port=80, host='0.0.0.0', threaded=False)